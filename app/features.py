# features.py — engineered features for the app (must match training)
# ------------------------------------------------------------------------------------
# IMPORTANT: paste the exact implementations from your notebook/training code.
#            This must include all geometry & skin EPC functions and their orchestrators.

import numpy as np, cv2
from typing import Dict, Any

# If you used skimage for EPCs, import here (safe to remove if unused)
try:
    from skimage import color, filters
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
except Exception:
    pass

# ============== PASTE YOUR FUNCTIONS BELOW (exactly as in training) ==============
# Examples of what to paste (names you used earlier):
# - compute_all_metrics(bgr_image, landmarks_flat)
# - compute_all_skin_epcs(cheek_patch_rgb)      # if you had a separate EPC function
# - any helpers: symmetry, fWHR, thirds, texture_roughness, skin_tone_deltaE_mean, etc.
# - ensure names and output dict keys match those in feature_order.json
# ------------------------------------------------------------------------------
# >>> PASTE: your exact code that computes engineered features here. <<<


# ============== Wrapper called by the app =======================================
def compute_all_features(image_rgb: np.ndarray, landmarks_flat: np.ndarray) -> Dict[str, Any]:
    """
    Returns a dict with EVERY engineered feature expected by the model.
    Keys MUST match those in app_artifacts/feature_order.json.
    """
    # Your training orchestrator took BGR:
    feats = compute_all_metrics(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), landmarks_flat)

    # If you computed extra EPCs in a separate function, merge them:
    # feats.update(compute_all_skin_epcs(image_rgb))

    # Any post-processing you applied before modeling (e.g., quads already included in training)
    return feats
