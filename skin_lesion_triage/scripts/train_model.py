"""
train_model.py
Two-phase transfer learning training script for the skin lesion triage model.
Designed to run on Google Colab (GPU runtime) but works locally too if you
have a GPU and the data/processed folder already populated by prepare_data.py.

Phase 1 (feature extraction): base CNN frozen, train only the new
  classification head. Fast, prevents destroying pretrained features early.
Phase 2 (fine-tuning): unfreeze the top N layers of the base CNN and
  train at a much lower learning rate to adapt features to skin lesions.

USAGE (Colab):
    !python train_model.py --arch mobilenetv2 --epochs_head 10 --epochs_finetune 15
    !python train_model.py --arch resnet50 --epochs_head 10 --epochs_finetune 15

Objective 2 target: validation recall >= 0.85 on the malignant class.
Recall is prioritized over raw accuracy because in a triage setting a missed
malignant case (false negative) is far costlier than a false positive.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision, AUC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

ARCH_BUILDERS = {
    "resnet50": (ResNet50, "resnet50_skin_lesion"),
    "mobilenetv2": (MobileNetV2, "mobilenetv2_skin_lesion"),
}

FINE_TUNE_UNFREEZE_LAYERS = {
    "resnet50": 30,       # unfreeze last 30 layers of ResNet50 in phase 2
    "mobilenetv2": 40,    # unfreeze last 40 layers of MobileNetV2 in phase 2
}


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED,
    )

    class_names = train_ds.class_names  # e.g. ['benign', 'malignant']
    print(f"[*] Class order (0/1): {class_names}")

    # Data augmentation applied only to training data — helps generalize to
    # the noisy, variably-lit "in the wild" smartphone images the proposal
    # targets, since HAM10000 itself is dermatoscopic (cleaner) imagery.
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
        layers.GaussianNoise(0.03),
    ])

    normalization = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_dir: Path) -> dict:
    """HAM10000 is heavily imbalanced (many more benign nv cases than
    malignant). Class weighting helps the model not just learn to always
    predict 'benign'."""
    benign_count = len(list((train_dir / "benign").glob("*.jpg")))
    malignant_count = len(list((train_dir / "malignant").glob("*.jpg")))
    total = benign_count + malignant_count
    weight_for_benign = total / (2.0 * benign_count)
    weight_for_malignant = total / (2.0 * malignant_count)
    print(f"[*] Class counts -> benign: {benign_count}, malignant: {malignant_count}")
    print(f"[*] Class weights -> benign: {weight_for_benign:.3f}, malignant: {weight_for_malignant:.3f}")
    return {0: weight_for_benign, 1: weight_for_malignant}


def build_model(arch: str):
    builder_fn, _ = ARCH_BUILDERS[arch]
    base_model = builder_fn(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", Recall(name="recall"), Precision(name="precision"), AUC(name="auc")],
    )


def tune_threshold(model, val_ds, target_recall: float = 0.85) -> dict:
    """
    Objective 2 asks for validation recall >= 0.85 on the malignant class.
    The default 0.5 cutoff on the sigmoid output rarely lands there directly,
    so this scans the validation set's precision/recall curve and picks the
    lowest threshold that still meets the recall target, preferring higher
    precision among threshold values that clear the bar. If no threshold on
    the curve reaches the target, it falls back to whichever threshold gives
    the highest recall available, and flags that fallback in the result.
    """
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).ravel()
    y_prob = model.predict(val_ds, verbose=0).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more point than thresholds; drop it
    # so the three arrays line up index for index.
    precision, recall = precision[:-1], recall[:-1]

    meets_target = recall >= target_recall
    if meets_target.any():
        candidate_idx = np.where(meets_target)[0]
        best_idx = candidate_idx[np.argmax(precision[candidate_idx])]
        met_target = True
    else:
        best_idx = int(np.argmax(recall))
        met_target = False

    chosen_threshold = float(thresholds[best_idx])
    return {
        "threshold": round(chosen_threshold, 4),
        "val_recall_at_threshold": round(float(recall[best_idx]), 4),
        "val_precision_at_threshold": round(float(precision[best_idx]), 4),
        "target_recall": target_recall,
        "met_target": met_target,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50/MobileNetV2 for skin lesion triage.")
    parser.add_argument("--arch", choices=["resnet50", "mobilenetv2"], default="mobilenetv2")
    parser.add_argument("--epochs_head", type=int, default=10, help="Phase 1: frozen-base epochs")
    parser.add_argument("--epochs_finetune", type=int, default=15, help="Phase 2: fine-tuning epochs")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _, model_filename = ARCH_BUILDERS[args.arch]
    checkpoint_path = MODEL_DIR / f"{model_filename}.keras"

    train_ds, val_ds, test_ds, class_names = build_datasets()
    class_weights = compute_class_weights(DATA_DIR / "train")

    model, base_model = build_model(args.arch)

    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_recall", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_recall", mode="max", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # ---------- Phase 1: train the head only ----------
    print("\n" + "=" * 60)
    print(f"PHASE 1: Training classification head ({args.arch}, base frozen)")
    print("=" * 60)
    compile_model(model, args.lr_head)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---------- Phase 2: fine-tune top layers of the base ----------
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tuning top layers of {args.arch}")
    print("=" * 60)
    base_model.trainable = True
    unfreeze_n = FINE_TUNE_UNFREEZE_LAYERS[args.arch]
    for layer in base_model.layers[:-unfreeze_n]:
        layer.trainable = False

    compile_model(model, args.lr_finetune)  # much lower LR to avoid catastrophic forgetting
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---------- Final evaluation on held-out test set ----------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION on test set")
    print("=" * 60)
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")

    if results.get("recall", 0) >= 0.85:
        print("\n[✓] Objective 2 target met: validation/test recall >= 0.85")
    else:
        print("\n[!] Recall below 0.85 target — consider: more epochs, stronger "
              "augmentation, adjusting class weights, or trying the other architecture.")

    # ---------- Threshold tuning on the validation set ----------
    print("\n" + "=" * 60)
    print("TUNING DECISION THRESHOLD on validation set")
    print("=" * 60)
    threshold_info = tune_threshold(model, val_ds, target_recall=0.85)
    threshold_path = MODEL_DIR / f"{model_filename}_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_info, f, indent=2)

    if threshold_info["met_target"]:
        print(f"[✓] Threshold {threshold_info['threshold']} reaches the 0.85 recall "
              f"target (val recall={threshold_info['val_recall_at_threshold']}, "
              f"val precision={threshold_info['val_precision_at_threshold']}).")
    else:
        print(f"[!] No threshold on the validation curve reached 0.85 recall. "
              f"Falling back to the highest-recall threshold available: "
              f"{threshold_info['threshold']} "
              f"(val recall={threshold_info['val_recall_at_threshold']}). "
              f"Consider retraining before trusting this model for triage.")
    print(f"    Saved to {threshold_path}")

    model.save(checkpoint_path)
    print(f"\n[✓] Model saved to {checkpoint_path}")
    print("    Copy this file and its *_threshold.json into the Flask app's /models folder to deploy it.")


if __name__ == "__main__":
    main()
