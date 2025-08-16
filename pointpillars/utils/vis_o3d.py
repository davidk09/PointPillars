# vis_o3d.py — headless & very verbose

import os, sys, cv2, numpy as np, inspect

def _log(*args): print("[vis_o3d]", *args, flush=True)

def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        _log(f"Created dir: {d}")
    if d:
        _log(f"Dir exists? {os.path.isdir(d)}  writeable? {os.access(d, os.W_OK)}")

def _probe_write(dir_path: str):
    try:
        test_path = os.path.join(dir_path, ".write_probe")
        with open(test_path, "wb") as f:
            f.write(b"ok")
        os.remove(test_path)
        _log(f"Probe write OK in: {dir_path}")
        return True
    except Exception as e:
        _log(f"Probe write FAILED in: {dir_path} -> {repr(e)}")
        return False

def _label_color(labels, i, default=(0, 0, 255)):
    if labels is None: return default
    try:
        if i >= len(labels): return default
        li = int(labels[i])
        COLORS_IMG = [[0,0,255],[0,255,0],[255,0,0],[0,255,255]]
        if 0 <= li < len(COLORS_IMG):
            c = COLORS_IMG[li]; return (int(c[0]), int(c[1]), int(c[2]))
        return default
    except Exception as e:
        _log(f"label color fallback for index {i}: {repr(e)}")
        return default

# Import helper
bbox3d2corners = None
try:
    from .process import bbox3d2corners as _bbox_rel
    bbox3d2corners = _bbox_rel
    _log("Imported bbox3d2corners via relative '.process'.")
except Exception as e_rel:
    _log("WARN: relative import failed:", repr(e_rel))
    try:
        from process import bbox3d2corners as _bbox_abs
        bbox3d2corners = _bbox_abs
        _log("Imported bbox3d2corners via absolute 'process'.")
    except Exception as e_abs:
        _log("WARN: absolute import failed:", repr(e_abs))

# Constants
LINES = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
]

# Import site diagnostics (where is this file, and who imports us)
try:
    _log(f"vis_o3d file: {__file__}")
    _log(f"sys.path[0]: {sys.path[0]}")
except Exception: pass

def _common_env_log():
    try:
        _log(f"CWD: {os.getcwd()}")
        try: _log(f"UID:GID = {os.getuid()}:{os.getgid()}")
        except Exception: pass
        _log(f"Python: {sys.version.split()[0]}")
    except Exception: pass

def vis_pc(pc, bboxes=None, labels=None, out_path=None):
    _log("=== vis_pc start ==="); _common_env_log()
    img_h, img_w = 800, 800
    vis_img = 255*np.ones((img_h,img_w,3), dtype=np.uint8)

    # Ranges (KITTI)
    x_min,x_max = 0.0,70.4
    y_min,y_max = -40.0,40.0
    sx = (img_w-1)/(x_max-x_min)
    sy = (img_h-1)/(y_max-y_min)

    # Points
    n_pts = 0
    if isinstance(pc, np.ndarray) and pc.ndim==2 and pc.shape[1]>=3 and pc.shape[0]>0:
        xs, ys = pc[:,0], pc[:,1]
        m = (xs>=x_min)&(xs<=x_max)&(ys>=y_min)&(ys<=y_max)
        xs, ys = xs[m], ys[m]
        xp = ((xs-x_min)*sx).astype(np.int32)
        yp = (img_h-1 - (ys-y_min)*sy).astype(np.int32)
        xp = np.clip(xp,0,img_w-1); yp = np.clip(yp,0,img_h-1)
        vis_img[yp, xp] = (0,255,0)
        n_pts = len(xp)
    _log(f"Points drawn: {n_pts} (input valid? {isinstance(pc,np.ndarray)})")

    # Boxes
    corners = None
    if bboxes is not None:
        bboxes = np.asarray(bboxes)
        _log(f"bboxes shape: {bboxes.shape}")
        if bboxes.ndim==2 and bboxes.shape[1]==7:
            if bbox3d2corners is None:
                _log("No bbox3d2corners available; skip boxes.")
            else:
                try:
                    corners = bbox3d2corners(bboxes)  # (n,8,3)
                    _log(f"Converted to corners: {corners.shape}")
                except Exception as e:
                    _log("Convert ERROR:", repr(e))
                    corners = None
        elif bboxes.ndim==3 and bboxes.shape[1]==8 and bboxes.shape[2]>=2:
            corners = bboxes
            _log("Using provided corners.")
        elif bboxes.ndim==2 and bboxes.shape==(8,3):
            corners = bboxes[None,...]
            _log("Single (8,3) corners → n=1.")
        else:
            _log("Unsupported bbox shape; skipping.")

    if corners is not None and isinstance(corners, np.ndarray) and corners.ndim==3 and corners.shape[1]==8:
        n_boxes = corners.shape[0]; _log(f"Drawing {n_boxes} boxes.")
        for i in range(n_boxes):
            box = corners[i]; pts_xy = box[:,:2].astype(np.float32)
            px = ((pts_xy[:,0]-x_min)*sx); py = (img_h-1 - (pts_xy[:,1]-y_min)*sy)
            pts_px = np.stack([px,py],axis=1).astype(np.int32)
            color = _label_color(labels,i)
            for a,b in LINES:
                cv2.line(vis_img, tuple(pts_px[a]), tuple(pts_px[b]), color, 2)
            for p in pts_px:
                cv2.circle(vis_img, tuple(p), 2, (0,0,0), -1)
    else:
        if bboxes is not None: _log("Boxes provided but not drawable.")

    # Save
    save_path = out_path if out_path else os.path.join("figures","vis_pc.png")
    abs_path = os.path.abspath(save_path)
    _ensure_dir_for(abs_path)
    _probe_write(os.path.dirname(abs_path))
    _log(f"Saving BEV to: {abs_path}")
    ok = False
    try:
        ok = cv2.imwrite(abs_path, vis_img)
    except Exception as e:
        _log("cv2.imwrite threw:", repr(e))
    _log(f"cv2.imwrite ok={ok}")
    if not ok:
        # Try a second fallback path for diagnostics (only logs, doesn’t change API)
        fallback = "/workspace/PointPillars/figures/vis_pc_fallback.png"
        try:
            _ensure_dir_for(fallback)
            _probe_write(os.path.dirname(fallback))
            ok2 = cv2.imwrite(fallback, vis_img)
            _log(f"Fallback save -> {fallback} ok={ok2}")
        except Exception as e:
            _log("Fallback failed:", repr(e))
    _log("=== vis_pc end ===")

def vis_img_3d(img, image_points, labels=None, rt=True, out_path=None):
    _log("=== vis_img_3d start ==="); _common_env_log()
    if img is None or not isinstance(img, np.ndarray):
        _log("ERROR: img is None or not ndarray"); return None
    H,W = img.shape[:2]; _log(f"Image: {img.shape} {img.dtype}")
    vis = img.copy()

    boxes_n = 0
    if image_points is None:
        _log("No image_points; saving raw image.")
    else:
        pts = np.asarray(image_points)
        if pts.ndim==2 and pts.shape==(8,2): pts = pts[None,...]
        if not (pts.ndim==3 and pts.shape[1]==8 and pts.shape[2]==2):
            _log(f"Bad shape for image_points: {pts.shape}")
        else:
            boxes_n = pts.shape[0]; _log(f"Drawing {boxes_n} boxes.")
            for i in range(boxes_n):
                c2d = pts[i].astype(np.int32)
                col = _label_color(labels,i)
                for a,b in LINES:
                    cv2.line(vis, tuple(c2d[a]), tuple(c2d[b]), col, 2)
                for p in c2d:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    _log(f"Boxes drawn: {boxes_n}")

    save_path = out_path if out_path else os.path.join("figures","vis_img_3d.png")
    abs_path = os.path.abspath(save_path)
    _ensure_dir_for(abs_path)
    _probe_write(os.path.dirname(abs_path))
    _log(f"Saving IMG to: {abs_path}")
    ok = False
    try:
        ok = cv2.imwrite(abs_path, vis)
    except Exception as e:
        _log("cv2.imwrite threw:", repr(e))
    _log(f"cv2.imwrite ok={ok}")
    if not ok:
        fallback = "/workspace/PointPillars/figures/vis_img_3d_fallback.png"
        try:
            _ensure_dir_for(fallback)
            _probe_write(os.path.dirname(fallback))
            ok2 = cv2.imwrite(fallback, vis)
            _log(f"Fallback save -> {fallback} ok={ok2}")
        except Exception as e:
            _log("Fallback failed:", repr(e))
    _log("=== vis_img_3d end ===")
    return vis

# at import time, tell us exactly which vis_pc we're exporting and from where
try:
    _log(f"vis_pc defined in: {inspect.getsourcefile(vis_pc)}")
except Exception: pass
