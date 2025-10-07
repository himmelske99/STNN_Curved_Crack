import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.crack import detect_crack_skeleton

def overlay_crack_on_field(gt_disps, gt_coords, gt_masks,
                           pred_disps, pred_coords, pred_masks,
                           save_dir, which="pred"):
    """
    Если переданы словари — перебирает их и строит по каждому кадру.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(gt_disps)):
        gt_u, gt_v = gt_disps[i]
        pr_u, pr_v = pred_disps[i]
        mask = gt_masks[i]
        X, Y = gt_coords[i]

        if which == "gt":
            skel = detect_crack_skeleton(gt_u, gt_v, valid_mask=mask)
            field = gt_u
            title = f"GT_frame_{i:03d}"
        else:
            skel = detect_crack_skeleton(pr_u, pr_v, valid_mask=mask)
            field = pr_u
            title = f"Pred_frame_{i:03d}"

        plt.figure(figsize=(4, 4), dpi=150)
        plt.imshow(np.abs(field), cmap="viridis")
        plt.contour(skel, colors="r", linewidths=0.8)
        plt.title(title)
        plt.axis("off")

        fpath = os.path.join(save_dir, f"{title}.png")
        plt.savefig(fpath, dpi=200)
        plt.close()

        print(f"✅ Saved overlay: {fpath}")
