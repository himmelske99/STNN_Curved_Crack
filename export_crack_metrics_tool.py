# # tools/export_metrics_csv.py
# # -*- coding: utf-8 -*-
# """
# Экспорт метрик и длины трещины в CSV.
# Сохраняет по каждому кадру: SSIM, MSE, CrackLen_GT, CrackLen_Pred, CrackLen_Error.
# """

# import os
# import csv
# import numpy as np
# from src.utils.crack import crack_length_series

# def export_metrics_csv(
#     save_dir,
#     gt_disps,
#     gt_coords,
#     gt_masks,
#     pred_disps,
#     pred_coords,
#     pred_masks,
#     ssim_vals=None,
#     mse_vals=None,
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     csv_path = os.path.join(save_dir, "crack_metrics.csv")

#     gt_len, keys_gt = crack_length_series(gt_disps, gt_coords, gt_masks)
#     pr_len, keys_pr = crack_length_series(pred_disps, pred_coords, pred_masks)

#     # возможна разная длина массивов
#     n = min(len(gt_len), len(pr_len))
#     gt_len, pr_len = gt_len[:n], pr_len[:n]
#     crack_err = np.abs(gt_len - pr_len)

#     # метрики (если заданы списками)
#     if ssim_vals is not None and len(ssim_vals) >= n:
#         ssim_vals = ssim_vals[:n]
#     else:
#         ssim_vals = [np.nan] * n

#     if mse_vals is not None and len(mse_vals) >= n:
#         mse_vals = mse_vals[:n]
#     else:
#         mse_vals = [np.nan] * n

#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Frame", "SSIM", "MSE", "CrackLen_GT(mm)", "CrackLen_Pred(mm)", "CrackLen_Error(mm)"])
#         for i in range(n):
#             writer.writerow([i, ssim_vals[i], mse_vals[i], gt_len[i], pr_len[i], crack_err[i]])

#     print(f"✅ Exported crack metrics to {csv_path}")
#     return csv_path





















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


