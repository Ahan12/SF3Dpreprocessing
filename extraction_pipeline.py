import os, glob, cv2, shutil
import numpy as np
from collections import Counter
from projection_utils import parse_pincam, parse_traj, project_points

# --- Configuration & Thresholds (Section 3.2) ---
MIN_DEPTH      = 0.10
THETA_DEPTH    = 0.05
THETA_POINTS   = 0.30
THETA_AREA     = 400
THETA_COLOR    = 0.30
MIN_COLOR_PTS  = 20
KEEP_TOP_VIEWS = 3
CROP_PAD_FRAC, CROP_PAD_MIN, MASK_DILATE = 1.2, 48, 7

FRAMES_DIR = './frames'
os.makedirs(FRAMES_DIR, exist_ok=True)

def intr_at(base, key):
    hits = glob.glob(f'{base}/hires_wide_intrinsics/*{key}*')
    return parse_pincam(hits[0]) if hits else (None, None, None)

def score_frame_cached(P, gray_pts, dep, gray_img, W, H, K, T):
    """Evaluates visibility, area, and color consistency for a given frame."""
    u, v, d = project_points(P, T, K)
    ok = (d > MIN_DEPTH) & (u>=0) & (u<W) & (v>=0) & (v<H)
    if ok.sum() == 0: return None
    ui, vi = u[ok].astype(int), v[ok].astype(int)

    rec  = dep[vi, ui]
    seen = (rec > 0) & (np.abs(d[ok]-rec) < THETA_DEPTH)
    vis  = seen.sum()/len(P)
    if vis < THETA_POINTS or seen.sum() < 2: return None

    uv, vv = u[ok][seen], v[ok][seen]
    area = max(uv.max()-uv.min(),1)*max(vv.max()-vv.min(),1)
    if area < THETA_AREA: return None

    corr = np.nan
    if gray_pts is not None and gray_img is not None and seen.sum() >= MIN_COLOR_PTS:
        ph, sc = gray_img[vi[seen], ui[seen]], gray_pts[ok][seen]
        if ph.std() > 1e-6 and sc.std() > 1e-6:
            corr = float(np.corrcoef(ph, sc)[0,1])
    if not np.isnan(corr) and corr < THETA_COLOR: return None
    
    return vis, float(area), corr

def build_view(P, base, key, T, cache_frame=False, visit=None, video=None):
    """Generates crop and pixel mask for a passing view."""
    W, H, K = intr_at(base, key)
    u, v, d = project_points(P, T, K)
    ok = (d > MIN_DEPTH) & (u>=0) & (u<W) & (v>=0) & (v<H)
    if ok.sum() == 0: return None
    
    x0, x1, y0, y1 = u[ok].min(), u[ok].max(), v[ok].min(), v[ok].max()
    px = max((x1-x0)*CROP_PAD_FRAC, CROP_PAD_MIN)
    py = max((y1-y0)*CROP_PAD_FRAC, CROP_PAD_MIN)
    X0, Y0 = int(max(0,x0-px)), int(max(0,y0-py))
    X1, Y1 = int(min(W,x1+px)), int(min(H,y1+py))
    if X1<=X0 or Y1<=Y0: return None

    src = glob.glob(f'{base}/hires_wide/*{key}*')[0]
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    
    m = np.zeros((Y1-Y0, X1-X0), np.uint8)
    m[np.clip(v[ok].astype(int)-Y0,0,Y1-Y0-1), np.clip(u[ok].astype(int)-X0,0,X1-X0-1)] = 1
    m = cv2.dilate(m, np.ones((MASK_DILATE,)*2, np.uint8)).astype(bool)

    fname = None
    if cache_frame:
        fname = f'{visit}_{video}_{key}.jpg'
        dst = f'{FRAMES_DIR}/{fname}'
        if not os.path.exists(dst): shutil.copy(src, dst)
        
    return dict(crop=img[Y0:Y1,X0:X1], mask=m, box=(X0,Y0,X1,Y1), 
                frame=key, frame_file=fname)
