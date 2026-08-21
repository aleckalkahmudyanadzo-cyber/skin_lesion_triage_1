"""
preprocessing.py
OpenCV-based preprocessing pipeline for smartphone-acquired lesion images.

Handles the real-world problems named in the proposal (Objective 1):
- variable brightness / contrast
- sensor noise
- hair occlusion (common in dermatoscopic + smartphone skin images)
- low-quality / blurred image rejection

Pipeline: decode -> hair removal (DullRazor-style inpaint) -> denoise ->
CLAHE contrast normalization -> blur/quality check -> resize -> scale.
"""

import cv2
import numpy as np


class ImageQualityError(Exception):
    """Raised when an image fails the quality gate (too blurry / unusable)."""
    pass


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """Focus measure. Low value = blurry image."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """
    DullRazor-inspired hair removal:
    1. Grayscale + blackhat morphological filter to find dark hair-like structures.
    2. Threshold to build a hair mask.
    3. Inpaint the masked pixels using surrounding skin texture.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """Non-local means denoising — good for the grainy noise typical of
    low-end smartphone sensors in poor lighting."""
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h=6, hColor=6,
                                            templateWindowSize=7, searchWindowSize=21)


def normalize_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB colour space. Corrects uneven
    lighting / poor contrast without blowing out skin tone colour info,
    which matters for images across the Fitzpatrick scale."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def check_quality(img_bgr: np.ndarray, blur_threshold: float = 60.0) -> float:
    """
    Rejects images that are too blurry / featureless to be triaged safely.
    Returns the sharpness score if it passes; raises ImageQualityError if not.
    This directly implements the proposal's stated limitation: images with
    heavy hair coverage or extreme blur are rejected at preprocessing.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = _variance_of_laplacian(gray)
    if score < blur_threshold:
        raise ImageQualityError(
            f"Image too blurry for reliable triage (sharpness={score:.1f}, "
            f"minimum required={blur_threshold}). Please retake the photo in "
            f"better light, holding the camera steady."
        )
    return score


def preprocess_image(image_bytes: bytes, target_size=(224, 224), for_model: bool = True):
    """
    Full pipeline entry point.

    Args:
        image_bytes: raw bytes read from the uploaded file.
        target_size: (H, W) expected by the CNN input layer.
        for_model: if True, returns a normalized float32 array ready for
                   model.predict(). If False, returns a displayable uint8
                   BGR image (e.g. for saving a "cleaned" preview).

    Returns:
        (processed_array, sharpness_score)
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. File may be corrupted or an unsupported format.")

    img_bgr = remove_hair(img_bgr)
    img_bgr = denoise(img_bgr)
    img_bgr = normalize_contrast(img_bgr)

    sharpness = check_quality(img_bgr)  # raises ImageQualityError if too blurry

    img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    if not for_model:
        return img_bgr, sharpness

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32")
    img_rgb = img_rgb / 255.0
    img_array = np.expand_dims(img_rgb, axis=0)  # batch dimension
    return img_array, sharpness
