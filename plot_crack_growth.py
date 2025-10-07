# # tools/plot_crack_growth.py
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from src.utils.crack import crack_length_series

# def plot_crack_growth(
#     gt_disps, gt_coords, gt_masks,
#     pred_disps, pred_coords, pred_masks, save_dir
# ):
#     """
#     Строит график роста трещины GT vs Pred по кадрам.
#     """
#     gt_len, gt_keys = crack_length_series(gt_disps, gt_coords, gt_masks)
#     pred_len, pred_keys = crack_length_series(pred_disps, pred_coords, pred_masks)

#     plt.figure(figsize=(6,4), dpi=150)
#     plt.plot(gt_keys, gt_len, 'o-', label="Ground Truth", lw=2)
#     plt.plot(pred_keys, pred_len, 's--', label="Predicted", lw=2)
#     plt.xlabel("Frame / Stage", fontsize=11)
#     plt.ylabel("Crack length (mm)", fontsize=11)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()

#     os.makedirs(save_dir, exist_ok=True)
#     fig_path = os.path.join(save_dir, "crack_length_comparison.png")

#     plt.savefig(fig_path, dpi=200)
#     plt.close()
#     print(f"✅ Crack growth plot saved to: {fig_path}")

















# tools/plot_crack_growth.py
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.crack import crack_length_series
from src.utils.crack import crack_length_series_compat as crack_length_series




def plot_crack_growth(gt_disps, gt_coords, gt_masks,
                      pred_disps, pred_coords, pred_masks, save_dir):
    gt_len, gt_keys = crack_length_series(gt_disps, gt_coords, gt_masks)
    pred_len, pred_keys = crack_length_series(pred_disps, pred_coords, pred_masks)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(gt_keys, gt_len, 'o-', label="Ground Truth", lw=2)
    plt.plot(pred_keys, pred_len, 's--', label="Predicted", lw=2)
    plt.xlabel("Frame / Stage", fontsize=11)
    plt.ylabel("Crack length (mm)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "crack_length_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✅ Crack growth plot saved to: {fig_path}")
