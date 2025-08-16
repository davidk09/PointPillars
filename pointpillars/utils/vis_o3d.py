# vis_o3d.py (headless / save-to-file, with debug logs)

import os
import sys
import cv2
import numpy as np

# ---------- Debug helpers ----------

def _log(*args):
    print("[vis_o3d]", *args, flush=True)

def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _label_color(labels, i, default=(0, 0, 255)):
    """Safely pick a color for labels[i]."""
    if labels is None:
        return default
    try:
        if i >= len(labels):
            return default
        li = int(labels[i])
        if 0 <= li < len(COLORS_IMG):
            c = COLORS_IMG[li]
            return (int(c[0]), int(c[1]), int(c[2]))
        return default
    except Exception as e:
        _log(f"label color fallback for index {i}: {repr(e)}")
        return default

# ---------- Try both relative and absolute imports for bbox3d2corners ----------

bbox3d2corners = None
try:
    from .process import bbox3d2corners as _bbox3d2corners_rel  # package context
    bbox3d2corners = _bbox3d2corners_rel
    _log("Imported bbox3d2corners via relative '.process'.")
except Exception as e_rel:
    _log("WARN: relative import '.process' failed:", repr(e_rel))
    try:
        from process import bbox3d2corners as _bbox3d2corners_abs  # module context
        bbox3d2corners = _bbox3d2corners_abs
        _log("Imported bbox3d2corners via absolute 'process'.")
    except Exception as e_abs:
        _log("WARN: absolute import 'process' failed:", repr(e_abs))
        _log("NOTE: (n,7) box -> corners conversion will be unavailable.")

# ---------- Constants ----------

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # unused but kept for compatibility
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]  # BGR

# 12 edges of a cuboid, assuming standard KITTI corner ordering
LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
]

# ---------- Functions ----------

def vis_pc(pc, bboxes=None, labels=None, out_path=None):
    """
    Project point cloud to a BEV-like image and save it to disk.

    Args:
        pc:         np.ndarray (N, >=3) with columns [x, y, z, ...] (lidar coords)
        bboxes:     optional, either (n,7) lidar boxes or (n,8,3) corners
        labels:     optional labels for boxes (used to color boxes)
        out_path:   path to save (default 'figures/vis_pc.png')

    Returns:
        None (writes file to disk). Check logs for the save location and result.
    """
    _log("=== vis_pc start ===")
    try:
        _log(f"CWD: {os.getcwd()}")
        try:
            _log(f"UID: {os.getuid()}  GID: {os.getgid()}")
        except Exception:
            pass
        _log(f"Python: {sys.version.split()[0]}")
    except Exception:
        pass

    img_h, img_w = 800, 800
    vis_img = 255 * np.ones((img_h, img_w, 3), dtype=np.uint8)

    # BEV projection ranges (meters)
    x_min, x_max = 0.0, 70.4
    y_min, y_max = -40.0, 40.0

    # pixel scale
    scale_x = (img_w - 1) / (x_max - x_min)
    scale_y = (img_h - 1) / (y_max - y_min)

    # Draw points
    if isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[1] >= 3 and pc.shape[0] > 0:
        xs = pc[:, 0]
        ys = pc[:, 1]
        x_mask = (xs >= x_min) & (xs <= x_max)
        y_mask = (ys >= y_min) & (ys <= y_max)
        mask = x_mask & y_mask
        xs, ys = xs[mask], ys[mask]
        x_pix = ((xs - x_min) * scale_x).astype(np.int32)
        y_pix = (img_h - 1 - (ys - y_min) * scale_y).astype(np.int32)
        x_pix = np.clip(x_pix, 0, img_w - 1)
        y_pix = np.clip(y_pix, 0, img_h - 1)
        for x, y in zip(x_pix, y_pix):
            vis_img[y, x] = (0, 255, 0)  # green
        _log(f"Points drawn: {len(x_pix)} / {pc.shape[0]}")
    else:
        _log("No valid point cloud provided; skipping points.")

    # Handle bboxes: accept (n,7) or (n,8,3)
    corners = None
    if bboxes is not None:
        bboxes = np.asarray(bboxes)
        _log(f"bboxes provided with shape: {bboxes.shape}")
        if bboxes.ndim == 2 and bboxes.shape[1] == 7:
            if bbox3d2corners is None:
                _log("ERROR: bbox3d2corners unavailable; cannot convert (n,7) boxes to corners.")
            else:
                try:
                    corners = bbox3d2corners(bboxes)  # (n,8,3)
                    _log(f"Converted (n,7) -> corners shape: {None if corners is None else corners.shape}")
                except Exception as e:
                    _log("ERROR converting boxes to corners:", repr(e))
                    corners = None
        elif bboxes.ndim == 3 and bboxes.shape[1] == 8 and bboxes.shape[2] >= 2:
            corners = bboxes
            _log("Using provided (n,8,3) corners.")
        elif bboxes.ndim == 2 and bboxes.shape == (8, 3):
            corners = bboxes[None, ...]
            _log("Interpreted single (8,3) corners array as 1 box.")
        else:
            _log("Unsupported bbox shape; skipping boxes.")

    # Draw boxes in BEV using projected corner XY (z ignored)
    if corners is not None and isinstance(corners, np.ndarray) and corners.ndim == 3 and corners.shape[1] == 8:
        n_boxes = corners.shape[0]
        _log(f"Drawing {n_boxes} boxes.")
        for i in range(n_boxes):
            box = corners[i]
            pts_xy = box[:, :2].astype(np.float32)
            px = ((pts_xy[:, 0] - x_min) * scale_x)
            py = (img_h - 1 - (pts_xy[:, 1] - y_min) * scale_y)
            pts_px = np.stack([px, py], axis=1).astype(np.int32)

            color = _label_color(labels, i)
            # Draw cuboid edges (projected into BEV)
            for (a, b) in LINES:
                p1 = tuple(pts_px[a])
                p2 = tuple(pts_px[b])
                cv2.line(vis_img, p1, p2, color, 2)
            # Mark corners
            for p in pts_px:
                cv2.circle(vis_img, tuple(p), 2, (0, 0, 0), -1)
    else:
        if bboxes is not None:
            _log("Boxes were provided but could not be interpreted; nothing drawn.")

    # Save
    save_path = out_path if out_path is not None else os.path.join("figures", "vis_pc.png")
    _ensure_dir_for(save_path)
    ok = cv2.imwrite(save_path, vis_img)
    _log(f"Saved BEV image -> '{save_path}' (ok={ok})")
    if not ok:
        _log("ERROR: cv2.imwrite returned False — path unwritable or invalid. "
             "Try an absolute path or a writable mount.")
    _log("=== vis_pc end ===")


def vis_img_3d(img, image_points, labels=None, rt=True, out_path=None):
    """
    Draw projected 3D bounding boxes on a camera image and save to disk.

    Args:
        img:           (H,W,3) numpy array (BGR, uint8 recommended)
        image_points:  (n,8,2) array (pixels) OR a single (8,2) array
        labels:        optional labels for coloring (len >= n)
        rt:            kept for API compatibility; unused here
        out_path:      path to save (default 'figures/vis_img_3d.png')

    Returns:
        Annotated image (np.ndarray). Also writes to disk.
    """
    _log("=== vis_img_3d start ===")
    if img is None or not isinstance(img, np.ndarray):
        _log("ERROR: 'img' is None or not a numpy array.")
        return None

    H, W = img.shape[:2]
    _log(f"Input image shape: {img.shape}, dtype={img.dtype}")

    vis_img = img.copy()

    if image_points is None:
        _log("No 'image_points' provided; saving original image.")
        boxes_n = 0
    else:
        pts = np.asarray(image_points)
        if pts.ndim == 2 and pts.shape == (8, 2):
            pts = pts[None, ...]  # single box
        if not (pts.ndim == 3 and pts.shape[1] == 8 and pts.shape[2] == 2):
            _log(f"ERROR: 'image_points' has unexpected shape {pts.shape}; expected (n,8,2) or (8,2).")
            boxes_n = 0
        else:
            boxes_n = pts.shape[0]
            _log(f"Drawing {boxes_n} projected boxes.")
            for i in range(boxes_n):
                corners_2d = pts[i].astype(np.int32)
                color = _label_color(labels, i)
                # Draw edges
                for (a, b) in LINES:
                    p1 = tuple(corners_2d[a])
                    p2 = tuple(corners_2d[b])
                    cv2.line(vis_img, p1, p2, color, 2)
                # Mark corners
                for p in corners_2d:
                    cv2.circle(vis_img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    _log(f"Boxes drawn on image: {boxes_n}")

    save_path = out_path if out_path is not None else os.path.join("figures", "vis_img_3d.png")
    _ensure_dir_for(save_path)
    ok = cv2.imwrite(save_path, vis_img)
    _log(f"Saved 3D overlay image -> '{save_path}' (ok={ok})")
    if not ok:
        _log("ERROR: cv2.imwrite returned False — path unwritable or invalid. "
             "Try an absolute path or a writable mount.")
    _log("=== vis_img_3d end ===")
    return vis_img
