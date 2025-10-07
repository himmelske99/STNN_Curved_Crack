# tools/export_metrics_csv.py
import os
import csv
import numpy as np
from src.utils.crack import crack_length_series

def export_metrics_csv(gt_disps, gt_coords, gt_masks,
                       pred_disps, pred_coords, pred_masks,
                       save_dir, fname="crack_metrics.csv", **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    gt_len, gt_keys = crack_length_series(gt_disps, gt_coords, gt_masks)
    pr_len, pr_keys = crack_length_series(pred_disps, pred_coords, pred_masks)

    # unify keys (outer join on keys)
    all_keys = sorted(set(gt_keys) | set(pr_keys))
    gt_map = {k: float(gt_len[i]) for i, k in enumerate(gt_keys)}
    pr_map = {k: float(pr_len[i]) for i, k in enumerate(pr_keys)}

    csv_path = os.path.join(save_dir, fname)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "gt_length_mm", "pred_length_mm"])
        for k in all_keys:
            w.writerow([k, gt_map.get(k, np.nan), pr_map.get(k, np.nan)])

    print(f"✅ Exported crack metrics to {csv_path}")


