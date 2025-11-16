# streamlit_app.py
import os, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

# ML
import joblib
from catboost import CatBoostRegressor, Pool

# ViT
import torch
import timm
from torchvision import transforms

# Face processing
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

st.set_page_config(page_title="Facial Attractiveness Analyzer (XAI)", layout="wide")
warnings.filterwarnings("ignore")

# -------------------- Relative paths --------------------
APP_DIR = Path(__file__).resolve().parent
ART_DIR = APP_DIR / "app_artifacts"

CB_PATH   = ART_DIR / "cb_full.cbm"
VIT_LIN   = ART_DIR / "vit_linear.joblib"
META_PATH = ART_DIR / "meta_model.joblib"
ORD_PATH  = ART_DIR / "feature_order.json"
PCA_PATH  = ART_DIR / "embed_scalers_pca_ga_v2.joblib"  # {"emb_scaler","pca_full"}

# -------------------- Load artifacts --------------------
@st.cache_resource
def load_artifacts():
    cb = CatBoostRegressor()
    cb.load_model(str(CB_PATH))
    vit_linear = joblib.load(VIT_LIN)
    meta = joblib.load(META_PATH)
    obj = joblib.load(PCA_PATH)
    emb_scaler, pca_full = obj["emb_scaler"], obj["pca_full"]
    with open(ORD_PATH, "r") as f:
        ord_info = json.load(f)
    feat_order = ord_info["engineered_feature_order"]
    m = re.match(r"^vit\[(\d+),([A-Za-z0-9_]+)\]$", ord_info["visual_label"])
    pca_dim = int(m.group(1)); vmodel = m.group(2).lower()
    return cb, vit_linear, meta, emb_scaler, pca_full, feat_order, pca_dim, vmodel

CB, VIT_LIN, META, EMB_SCALER, PCA_FULL, FEAT_ORDER, P_DIM, V_MODEL = load_artifacts()

# -------------------- ViT (frozen, pooled) --------------------
@st.cache_resource
def load_vit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0, global_pool="avg")
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),            # HWC [0,255] RGB -> CHW [0,1]
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, preprocess, device

VIT, VIT_PREP, DEVICE = load_vit()

def vit_embed_rgb(rgb):
    with torch.no_grad():
        x = VIT_PREP(rgb)       # CHW
        x = x.unsqueeze(0).to(DEVICE)
        feat = VIT(x).cpu().numpy()
    return feat.reshape(-1)      # 768-d

# -------------------- Preprocess: mask → align → CLAHE → final landmarks --------------------
FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
SEG      = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

def preprocess_align(image_rgb: np.ndarray):
    seg_res = SEG.process(image_rgb)
    cond = np.stack((seg_res.segmentation_mask,)*3, axis=-1) > 0.1
    masked = np.where(cond, image_rgb, 0).astype(np.uint8)

    h, w, _ = masked.shape
    r0 = FACE_MESH.process(masked)
    if not r0.multi_face_landmarks:
        return None, "No face landmarks detected."

    lm0 = np.array([(lm.x * w, lm.y * h) for lm in r0.multi_face_landmarks[0].landmark])
    left_eye  = lm0[[33,133]].mean(axis=0)
    right_eye = lm0[[362,263]].mean(axis=0)
    dY, dX = right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    center = ((left_eye[0]+right_eye[0])//2, (left_eye[1]+right_eye[1])//2)
    rotM = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(masked, rotM, (w, h))

    r1 = FACE_MESH.process(aligned)
    if not r1.multi_face_landmarks:
        return None, "No face landmarks after alignment."

    lm1 = np.array([(lm.x * w, lm.y * h) for lm in r1.multi_face_landmarks[0].landmark])
    x_min, y_min = lm1.min(axis=0); x_max, y_max = lm1.max(axis=0)
    pad_x = (x_max-x_min)*0.30; pad_y = (y_max-y_min)*0.30
    x1,y1 = int(max(0, x_min-pad_x)), int(max(0, y_min-pad_y))
    x2,y2 = int(min(w, x_max+pad_x)), int(min(h, y_max+pad_y))
    crop = aligned[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "Invalid crop."

    crop = cv2.resize(crop, (512,512), interpolation=cv2.INTER_AREA)
    lab  = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    norm = cv2.cvtColor(cv2.merge((Lc,A,B)), cv2.COLOR_LAB2RGB)

    r2 = FACE_MESH.process(norm)
    if not r2.multi_face_landmarks:
        return None, "No face landmarks on normalized crop."

    lm2 = np.array([[lm.x, lm.y, lm.z] for lm in r2.multi_face_landmarks[0].landmark]).flatten()
    return dict(processed_image=norm, landmarks=lm2), None

# -------------------- Engineered features --------------------
# Canonical indices
IDX = {
    "nose_tip": 1, "chin": 152, "left_eye_outer": 33, "right_eye_outer": 263,
    "left_mouth_corner": 61, "right_mouth_corner": 291,
    "left_eyebrow": [70,63,105,66,107,55,65,52,53,46],
    "right_eyebrow": [336,296,334,293,300,283,282,295,285,276],
    "left_eye_outline": [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    "right_eye_outline": [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
    "nose_bridge": [6,197,195,5,4,1,19,94],
    "nose_tip_region": [4,1,2,98,327],
    "mouth_outer": [61,146,91,181,84,17,314,405,321,375,291,308],
    "forehead_landmarks": [103,104,69,105,107,10],
    # cheek polygon used for EPCs
    "cheek_region": [234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356],
}

def region_centroid(land, idxs):
    return land[idxs, :2].mean(axis=0)

def euclid(a, b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))

def angle_between(a, b, c):
    a,b,c = np.array(a),np.array(b),np.array(c)
    ba, bc = a-b, c-b
    denom = (np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cosang = np.dot(ba, bc)/denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def face_bounds(land):
    xs, ys = land[:,0], land[:,1]
    return {"minx": xs.min(), "maxx": xs.max(), "miny": ys.min(), "maxy": ys.max(),
            "width": xs.max()-xs.min(), "height": ys.max()-ys.min()}

def symmetry_score(land, bounds):
    mid_x = (bounds["minx"] + bounds["maxx"]) / 2.0
    left_pts  = land[land[:,0] <  mid_x][:,:2]
    right_pts = land[land[:,0] >= mid_x][:,:2]
    if left_pts.size==0 or right_pts.size==0: return np.nan
    left_ref = left_pts.copy(); left_ref[:,0] = 2*mid_x - left_ref[:,0]
    d = [np.linalg.norm(right_pts - p, axis=1).min() for p in left_ref]
    return max(0.0, 1.0 - (np.mean(d) / (bounds["width"] + 1e-9)))

def thirds(land, bounds):
    top_y = bounds['miny']
    brow_y = region_centroid(land, IDX['left_eyebrow'] + IDX['right_eyebrow'])[1]
    nose_base_y = region_centroid(land, IDX['nose_tip_region'])[1]
    chin_y = land[IDX['chin'], 1]
    upper  = brow_y - top_y
    middle = nose_base_y - brow_y
    lower  = chin_y - nose_base_y
    H = bounds['height'] + 1e-9
    return {"upper_third_ratio": upper/H, "middle_third_ratio": middle/H, "lower_third_ratio": lower/H}

def goldenish(land, bounds):
    face_h, face_w = bounds['height'], bounds['width']
    lc = region_centroid(land, IDX['left_eye_outline'])
    rc = region_centroid(land, IDX['right_eye_outline'])
    interocular = euclid(lc, rc)
    eye_w = euclid(land[33,:2], land[133,:2])
    return {
        "face_len_width_ratio": face_h/(face_w+1e-9),
        "eye_interocular_ratio": eye_w/(interocular+1e-9)
    }

def mandibular_angle(land):
    mid_y = np.median(land[:,1])
    lower = land[land[:,1] > mid_y][:,:2]
    if lower.shape[0] < 3: return {"mandibular_angle_deg": np.nan}
    left_jaw  = lower[np.argmin(lower[:,0])]
    right_jaw = lower[np.argmax(lower[:,0])]
    chin = land[IDX['chin'], :2]
    return {"mandibular_angle_deg": angle_between(left_jaw, chin, right_jaw)}

def cheek_width_ratio(land, bounds):
    mid_y = (bounds['miny']+bounds['maxy'])/2.0
    band_h = bounds['height']*0.18
    band = land[(land[:,1]>(mid_y-band_h)) & (land[:,1]<(mid_y+band_h))][:,:2]
    if band.shape[0] < 2: return {"bizygomatic_width_ratio": np.nan}
    left = band[np.argmin(band[:,0])]; right = band[np.argmax(band[:,0])]
    width = euclid(left, right)
    return {"bizygomatic_width_ratio": width/(bounds["width"]+1e-9)}

def eyes_metrics(land, bounds):
    lc = region_centroid(land, IDX['left_eye_outline'])
    rc = region_centroid(land, IDX['right_eye_outline'])
    interpup = euclid(lc, rc)
    L = land[IDX['left_eye_outline'], :2]; R = land[IDX['right_eye_outline'], :2]
    Lh = L[:,1].max()-L[:,1].min(); Lw = L[:,0].max()-L[:,0].min()
    Rh = R[:,1].max()-R[:,1].min(); Rw = R[:,0].max()-R[:,0].min()
    mean_iris_exp = 0.5*((Lh/(Lw+1e-9)) + (Rh/(Rw+1e-9)))
    return {
        "interpup_face_width_ratio": interpup/(bounds['width']+1e-9),
        "mean_iris_exposure": mean_iris_exp
    }

def nose_angles(land):
    nb = region_centroid(land, IDX['nose_bridge'])
    nt = region_centroid(land, IDX['nose_tip_region'])
    forehead = land[10,:2]
    mouth_mid = region_centroid(land, IDX['mouth_outer'])
    return {
        "nasofrontal_deg": angle_between(forehead, nb, nt),
        "nasolabial_deg":  angle_between(nb, nt, mouth_mid)
    }

def fwh_ratio(land, bounds):
    cz = cheek_width_ratio(land, bounds)["bizygomatic_width_ratio"] * bounds['width']
    brow_c = region_centroid(land, IDX['left_eyebrow'] + IDX['right_eyebrow'])
    upper_lip_y = land[13,1]
    upper_face_h = upper_lip_y - brow_c[1]
    return {"fWHR": cz/(upper_face_h+1e-9)}

def lip_metrics(land):
    Lc = land[IDX['left_mouth_corner'], :2]; Rc = land[IDX['right_mouth_corner'], :2]
    w = euclid(Lc, Rc)
    mouth = land[IDX['mouth_outer'], :2]
    h = mouth[:,1].max() - mouth[:,1].min()
    return {"lip_width_px": w, "mouth_height_px": h}

# ---- Skin EPC helpers
from skimage import color, filters
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def cheek_patch_rgb(image_rgb, land):
    h, w, _ = image_rgb.shape
    poly = (land[IDX['cheek_region'], :2] * [w, h]).astype(int)
    x,y,ww,hh = cv2.boundingRect(poly)
    patch = image_rgb[y:y+hh, x:x+ww]
    return patch

def skin_color_basic(bgr, land, bounds):
    h,w,_ = bgr.shape
    x0,x1 = int(max(0,bounds['minx'])), int(min(w,bounds['maxx']))
    y0,y1 = int(max(0,bounds['miny'])), int(min(h,bounds['maxy']))
    crop = bgr[y0:y1, x0:x1]
    if crop.size < 10:
        return {"L_variance": np.nan, "a_mean_redness": np.nan, "b_mean_yellowness": np.nan, "texture_roughness": np.nan}
    lab = color.rgb2lab(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    return {
        "L_variance": float(np.var(lab[:,:,0])),
        "a_mean_redness": float(np.mean(lab[:,:,1])),
        "b_mean_yellowness": float(np.mean(lab[:,:,2])),
        "texture_roughness": float(np.var(lbp)),
    }

def epc_tone_evenness(patch_rgb):
    if patch_rgb.size == 0: return {}
    lab = color.rgb2lab(patch_rgb)
    l,a,b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    mu = [np.mean(l), np.mean(a), np.mean(b)]
    deltaE = np.sqrt(np.mean((l-mu[0])**2 + (a-mu[1])**2 + (b-mu[2])**2))
    return {
        'skin_tone_L_var': float(np.var(l)),
        'skin_tone_a_var': float(np.var(a)),
        'skin_tone_b_var': float(np.var(b)),
        'skin_tone_deltaE_mean': float(deltaE),
    }

def epc_surface_evenness(patch_rgb):
    if patch_rgb.size == 0: return {}
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    smooth = filters.gaussian(gray, sigma=3)
    Ra = np.mean(np.abs(gray - smooth))
    return {
        'skin_surface_contrast': float(graycoprops(glcm, 'contrast')[0,0]),
        'skin_surface_homogeneity': float(graycoprops(glcm, 'homogeneity')[0,0]),
        'skin_surface_correlation': float(graycoprops(glcm, 'correlation')[0,0]),
        'skin_surface_entropy': float(-(glcm*np.log2(glcm + 1e-9)).sum()),
        'skin_surface_Ra_roughness': float(Ra),
    }

def epc_firmness(patch_rgb):
    if patch_rgb.size == 0: return {}
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-9)
    return {
        'skin_firmness_laplacian_var': float(lap_var),
        'skin_firmness_edge_density': float(edge_density),
    }

def epc_glow(patch_rgb):
    if patch_rgb.size == 0: return {}
    lab = color.rgb2lab(patch_rgb)
    L = lab[:,:,0]
    th = np.percentile(L, 90)
    spec_ratio = np.sum(L > th) / (L.size + 1e-9)
    return {
        'skin_glow_mean_L': float(np.mean(L)),
        'skin_glow_L_var_inv': float(1.0/(np.var(L)+1e-6)),
        'skin_glow_specular_ratio': float(spec_ratio),
    }

def forehead_width_ratio(land, bounds):
    fw = land[IDX['forehead_landmarks'],0]
    return {"forehead_width_ratio": (fw.max() - fw.min())/(bounds['width']+1e-9)}

def brow_prominence(land):
    bz = land[IDX['left_eyebrow'] + IDX['right_eyebrow'], 2].mean()
    ez = land[IDX['left_eye_outline'] + IDX['right_eye_outline'], 2].mean()
    return {"brow_prominence_z": float(bz - ez)}

def eyelid_thickness(land):
    brow_y = region_centroid(land, IDX['left_eyebrow'] + IDX['right_eyebrow'])[1]
    upper_eyelid_y = land[[159,386], 1].mean()
    return {"eyelid_thickness_proxy": float(upper_eyelid_y - brow_y)}

def hair_luminance(image_bgr, land):
    fy = land[10,1]
    if fy < 20: return {"hair_luminance": np.nan}
    roi = image_bgr[0:int(fy)-10, :]
    if roi.size < 10: return {"hair_luminance": np.nan}
    return {"hair_luminance": float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())}

def compute_all_metrics(image_bgr: np.ndarray, landmarks_flat: np.ndarray) -> dict:
    land = landmarks_flat.reshape(478,3).copy()
    h,w,_ = image_bgr.shape
    land[:,0] *= w; land[:,1] *= h; land[:,2] *= w
    bounds = face_bounds(land)

    out = {}
    out['symmetry_score'] = symmetry_score(land, bounds)
    out.update( thirds(land, bounds) )
    out.update( goldenish(land, bounds) )
    out.update( mandibular_angle(land) )
    out.update( cheek_width_ratio(land, bounds) )
    out.update( eyes_metrics(land, bounds) )
    out.update( nose_angles(land) )
    out.update( fwh_ratio(land, bounds) )
    out.update( lip_metrics(land) )
    out.update( skin_color_basic(image_bgr, land, bounds) )
    out.update( forehead_width_ratio(land, bounds) )
    out.update( brow_prominence(land) )
    out.update( eyelid_thickness(land) )
    out.update( hair_luminance(image_bgr, land) )
    # EPCs on cheek patch (RGB)
    patch = cheek_patch_rgb(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), land / [w,h,w])  # land normalized for ROI indexing above
    out.update( epc_tone_evenness(patch) )
    out.update( epc_surface_evenness(patch) )
    out.update( epc_firmness(patch) )
    out.update( epc_glow(patch) )
    return out

def compute_all_features(image_rgb: np.ndarray, landmarks_flat: np.ndarray) -> dict:
    # image_bgr for geometric + basic skin; EPCs use RGB internally
    feats = compute_all_metrics(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), landmarks_flat)
    return feats

def ensure_column_order(feat_dict: dict, order: list) -> pd.DataFrame:
    row = {k: np.nan for k in order}
    for k,v in feat_dict.items():
        if k in row: row[k] = v
    return pd.DataFrame([row])[order]

# -------------------- UI --------------------
st.title("Facial Attractiveness Analyzer — Explainable (Research Prototype)")
st.caption("Educational prototype. Uses engineered facial metrics + ViT visual embeddings + linear meta for a transparent score on [1,10].")

with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("Upload a clear, front-facing JPG/PNG", type=["jpg","jpeg","png"])
    domain = st.selectbox("Domain prior (dataset offset)", ["SCUT-like (default)","CFD-like"])
    st.markdown("**Privacy:** images are processed in-memory and not stored.")
    st.markdown("**Tip:** good lighting, frontal pose, minimal occlusion.")

c1, c2 = st.columns(2)

if file is not None:
    pil = Image.open(file).convert("RGB")
    rgb = np.array(pil)

    with st.spinner("Preprocessing (mask, align, normalize, landmarks)…"):
        out, err = preprocess_align(rgb)
    if err:
        st.error(err); st.stop()

    proc = out["processed_image"]
    lms  = out["landmarks"]

    with c1: st.image(pil, caption="Original", use_column_width=True)
    with c2: st.image(proc, caption="Processed", use_column_width=True)

    with st.spinner("Extracting engineered features…"):
        feats = compute_all_features(proc, lms)
    X_eng_one = ensure_column_order(feats, FEAT_ORDER)

    # EPC (CatBoost)
    epc_pred = float(CB.predict(X_eng_one)[0])

    # Visual (ViT → scaler → PCA → linear)
    with st.spinner("Computing ViT embedding…"):
        feat_768 = vit_embed_rgb(proc)
    E_s = EMB_SCALER.transform(feat_768.reshape(1,-1))
    E_p = PCA_FULL.transform(E_s)[:, :P_DIM]
    vit_pred = float(VIT_LIN.predict(E_p)[0])

    # Meta
    srcCFD = 1.0 if domain == "CFD-like" else 0.0
    X_meta = np.array([[epc_pred, vit_pred, srcCFD]], dtype=float)
    final_score = float(META.predict(X_meta)[0])
    score_1_10 = max(1.0, min(10.0, final_score))

    st.subheader(f"Attractiveness Score: **{score_1_10:.2f}/10**")
    st.write(f"Meta inputs → EPC={epc_pred:.3f}, ViT={vit_pred:.3f}, srcCFD={srcCFD:.1f}")

    # Explainability: CatBoost SHAP for engineered features
    with st.spinner("Computing feature contributions (SHAP)…"):
        shap_vals = CB.get_feature_importance(Pool(X_eng_one), type="ShapValues")
        shap_sample = shap_vals[0]
        bias = shap_sample[-1]
        contribs = (
            pd.DataFrame({"feature": FEAT_ORDER, "contribution": shap_sample[:-1]})
              .assign(abs_contrib=lambda d: d["contribution"].abs())
              .sort_values("abs_contrib", ascending=False)
        )

    st.markdown("### Engineered Feature Contributions (sorted by |impact|)")
    st.dataframe(contribs.drop(columns=["abs_contrib"]), use_container_width=True, height=420)
    st.bar_chart(contribs.set_index("feature")["contribution"])

    report = {
        "score_out_of_10": score_1_10,
        "meta_inputs": {"epc_pred": epc_pred, "vit_pred": vit_pred, "srcCFD": srcCFD},
        "engineered_features": feats,
        "engineered_contributions": dict(zip(contribs["feature"], contribs["contribution"])),
        "bias": float(bias)
    }
    st.download_button("Download JSON report", data=json.dumps(report, indent=2),
                       file_name="explainability_report.json", mime="application/json")
else:
    st.info("Upload a frontal, well-lit image to begin.")
