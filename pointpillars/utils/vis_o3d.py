import cv2
import numpy as np
import os
from .process import bbox3d2corners  # convert (n,7) -> (n,8,3) when needed


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]


def vis_pc(pc, bboxes=None, labels=None, out_path=None):
    """
    Project point cloud to a BEV-like image and save it.
    - pc: np.ndarray (N, >=3) with columns [x, y, z, ...] (lidar coords)
    - bboxes: optional, either (n,7) lidar boxes or (n,8,3) corners
    - labels: optional labels for boxes (used to color boxes)
    - out_path: optional path under which to save the image. If None saves to figures/vis_pc.png
    """
    img_h, img_w = 800, 800
    vis_img = 255 * np.ones((img_h, img_w, 3), dtype=np.uint8)

    # BEV projection ranges (meters)
    x_min, x_max = 0.0, 70.4
    y_min, y_max = -40.0, 40.0

    # pixel scale
    scale_x = (img_w - 1) / (x_max - x_min)
    scale_y = (img_h - 1) / (y_max - y_min)

    # draw points
    if isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[1] >= 3 and len(pc) > 0:
        xs = pc[:, 0]
        ys = pc[:, 1]
        x_mask = (xs >= x_min) & (xs <= x_max)
        y_mask = (ys >= y_min) & (ys <= y_max)
        mask = x_mask & y_mask
        xs, ys = xs[mask], ys[mask]
        x_pix = ((xs - x_min) * scale_x).astype(np.int32)
        y_pix = (img_h - 1 - (ys - y_min) * scale_y).astype(np.int32)
        # clamp
        x_pix = np.clip(x_pix, 0, img_w - 1)
        y_pix = np.clip(y_pix, 0, img_h - 1)
        for x, y in zip(x_pix, y_pix):
            vis_img[y, x] = (0, 255, 0)  # green

    # handle bboxes: accept (n,7) or (n,8,3)
    corners = None
    if bboxes is not None:
        bboxes = np.asarray(bboxes)
        if bboxes.ndim == 2 and bboxes.shape[1] == 7:
            # convert to corners
            try:
                corners = bbox3d2corners(bboxes)  # (n,8,3)
            except Exception:
                corners = None
        elif bboxes.ndim == 3 and bboxes.shape[1] == 8 and bboxes.shape[2] >= 2:
            corners = bboxes

    if corners is not None:
        for i, box in enumerate(corners):
            pts = box[:, :2].copy()
            px = ((pts[:, 0] - x_min) * scale_x)
            py = (img_h - 1 - (pts[:, 1] - y_min) * scale_y)
            pts_px = np.stack([px, py], axis=1).astype(np.int32)
            pts_px = pts_px.reshape((-1, 1, 2))
            color = (int(COLORS_IMG[labels[i]][0]) if (labels is not None and 0 <= labels[i] < len(COLORS_IMG)) else 0,
                     int(COLORS_IMG[labels[i]][1]) if (labels is not None and 0 <= labels[i] < len(COLORS_IMG)) else 0,
                     int(COLORS_IMG[labels[i]][2]) if (labels is not None and 0 <= labels[i] < len(COLORS_IMG)) else 255)
            cv2.polylines(vis_img, [pts_px], isClosed=True, color=color, thickness=2)

    os.makedirs("figures", exist_ok=True)
    save_path = out_path if out_path is not None else os.path.join("figures", "vis_pc.png")
    cv2.imwrite(save_path, vis_img)


def vis_img_3d(img, image_points, labels=None, rt=True, out_path=None):
    """
    Draw projected 3D bounding boxes on camera image and save.
    - img: (H,W,3) numpy array (BGR)
    - image_points: (n,8,2) array of projected corners per box (in pixel coords)
    - labels: optional labels for coloring
    - rt: ignored for window display; kept for API compatibility
    - out_path: optional save path. If None saves to figures/vis_img_3d.png
    Returns annotated image.
    """
    if img is None:
        return None
    vis_img = img.copy()
    if isinstance(image_points, np.ndarray) and image_points.ndim == 3 and image_points.shape[1] == 8:
        for i in range(image_points.shape[0]):
            pts = image_points[i].astype(np.int32)
            # draw edges according to LINES
            for l in LINES:
                p1 = tuple(pts[l[0]])
                p2 = tuple(pts[l[1]])
                color = (int(COLORS_IMG[labels[i]][0]) if (labels is not None and 0 <= labels[i] < len(COLORS_IMG)) else 0,
                         int(COLORS_IMG[labels[i]][1]) if (labels is not None and 0 <= labels[i] < len(COLORS_IMG)) else 0,
                         int(COLORS_IMG[labels[i]][2]) if (labels is not None and 0 <= labels[i] < len(COLORS_IMG)) else 255)
                cv2.line(vis_img, p1, p2, color, 2)
            # draw corner points
            for pt in pts:
                cv2.circle(vis_img, tuple(int(x) for x in pt), radius=3, color=(0, 0, 255), thickness=-1)

    os.makedirs("figures", exist_ok=True)
    save_path = out_path if out_path is not None else os.path.join("figures", "vis_img_3d.png")
    cv2.imwrite(save_path, vis_img)
    return vis_img

