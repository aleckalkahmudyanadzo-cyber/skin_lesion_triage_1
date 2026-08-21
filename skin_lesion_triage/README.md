# Skin Lesion Triage System

A web-based deep learning triage tool for dermatological lesions captured on
smartphone cameras, built for a resource-limited clinical workflow context
(Midlands State University final-year project, HCSE236 / HCSCI240).

**This tool is a triage aid only — it does not provide a clinical diagnosis.**

## Project Structure

```
skin_lesion_triage/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # HTTP endpoints (/, /predict, /health, /stats)
│   ├── model_manager.py     # Model loading, inference, Grad-CAM
│   ├── preprocessing.py     # OpenCV pipeline (hair removal, denoise, CLAHE, quality gate)
│   ├── database.py          # SQLite usage logging
│   ├── static/{css,js,uploads}/
│   └── templates/{base.html, index.html}
├── scripts/
│   ├── prepare_data.py      # HAM10000 download + train/val/test split
│   └── train_model.py       # Two-phase transfer learning (Colab-ready)
├── models/                  # Trained .keras files go here
├── data/{raw,processed}/    # Dataset storage (gitignored)
├── instance/                # SQLite DB lives here
├── config.py
├── run.py
├── requirements.txt
├── setup.bat                # Windows one-click setup
└── README.md
```

## Quick Start (Windows)

1. Double-click `setup.bat` (or run it from cmd). This creates a virtual
   environment and installs everything in `requirements.txt`.
2. Get the dataset:
   ```
   python scripts\prepare_data.py --kaggle_download
   ```
   Requires a Kaggle API token at `C:\Users\<you>\.kaggle\kaggle.json`
   (get one from kaggle.com/settings -> API -> Create New Token).
3. Train a model — **strongly recommended on Google Colab** for free GPU:
   - Easiest path: open `colab/train_on_colab.ipynb` in Google Colab
     (colab.research.google.com -> Upload notebook), set Runtime -> GPU,
     and run every cell top to bottom. It uploads this whole project,
     downloads HAM10000 via Kaggle, preps the data, and trains both
     MobileNetV2 and ResNet50, then lets you download the `.keras` files.
   - Manual path: zip and upload the `scripts/` folder and `data/processed/`
     to Colab yourself, then run
     `!python train_model.py --arch mobilenetv2 --epochs_head 10 --epochs_finetune 15`
   - Either way, drop the resulting `.keras` file(s) into the local
     `models/` folder.
4. Run the app:
   ```
   python run.py
   ```
5. Open http://127.0.0.1:5000

## How It Works

1. **Upload** — user drags/drops or selects a JPG/PNG image.
2. **Preprocess** (`preprocessing.py`) — hair removal (DullRazor-style
   inpainting), non-local-means denoising, CLAHE contrast normalization,
   then a blur/quality gate rejects unusable images before they ever reach
   the model.
3. **Predict** (`model_manager.py`) — the chosen architecture
   (MobileNetV2 by default, ResNet50 optional) outputs a malignant-suspect
   probability.
4. **Explain** — Grad-CAM produces a heatmap over the region of the image
   that most influenced the prediction, overlaid for clinical
   interpretability.
5. **Log** — an anonymous (hashed, no PII) record of the prediction is
   stored in SQLite for later usability evaluation (SUS questionnaire,
   Objective 4).

## Switching Architectures

Set the `ACTIVE_MODEL` environment variable to `resnet50` or `mobilenetv2`
before running `run.py`, or edit the default in `config.py`. Both must be
trained separately via `train_model.py --arch <name>` and their `.keras`
files placed in `models/`.

## Known Limitations (per proposal)

- Triage only, not a diagnosis.
- Images with heavy hair coverage or extreme blur are rejected at the
  preprocessing stage rather than passed to the model.
- Inference latency on a free-tier cloud host is not yet benchmarked —
  measure this during deployment testing.
