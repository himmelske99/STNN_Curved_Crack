import os
import numpy as np
import torch
import matplotlib; matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import time
import io, contextlib

from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from src.deep_learning import nets, evaluate
from src.dataprocessing import transforms
from src.dataprocessing.dataset import Datasets
from src.utils.plot import (
    Plot_test_figures_Chinese,
    Plot_test_figures_Chinese_u_all,
    Plot_test_figures_IJF,
    record_and_plot_test_SSIM,
    record_and_plot_test_MSE,
)
from src.utils.utilityfunctions import Load_time_index, compute_border, crop_border_nd


def publication_panels(*args, **kwargs):
    print("ℹ️ publication_panels(): функция не реализована — пропуск.")



model_name = "N_type_test"
# Путь для сохранения результатов
SAVE_PATH = os.path.join("res", "OutputImagesAndModelsForResults", "evaluates", model_name, "results")
os.makedirs(SAVE_PATH, exist_ok=True)





# ===================== preview / saving control =====================
# Сохраняем картинки только для этих индексов (как было "1–2 изображения")
SAVE_ONLY_INDEXES = (0, 1)  # при желании: (0, 1, 80, 159)
# ===================================================================


def _calc_errors_silent(calc_fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return calc_fn(*args, **kwargs)


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            X, Y, M = batch
            return X, Y, M
        elif len(batch) == 2:
            X, Y = batch
            return X, Y, None
    # на всякий случай — стандартная попытка
    X, Y = batch
    return X, Y, None


def masked_mse_loss(pred, target, mask=None):
    device = pred.device
    target = target.to(device)
    if mask is not None:
        mask = mask.to(device)

    Cp = pred.size(2)
    Ct = target.size(2)
    C = Cp if Cp < Ct else Ct
    pred_c = pred[:, :, :C, ...]
    target_c = target[:, :, :C, ...]

    diff = (pred_c - target_c) ** 2

    if mask is None:
        return diff.mean()

    if mask.ndim == 4:  # [B,1,H,W] -> [B,1,1,H,W]
        mask = mask.unsqueeze(2)
    if mask.size(2) != 1:
        mask = mask[:, :, :1]

    T_pred = pred_c.size(1)
    if mask.size(1) > T_pred:
        mask = mask[:, :T_pred]
    elif mask.size(1) < T_pred:
        if mask.size(1) == 1:
            mask = mask.repeat(1, T_pred, 1, 1, 1)
        else:
            k = (T_pred + mask.size(1) - 1) // mask.size(1)
            mask = mask.repeat(1, k, 1, 1, 1)[:, :T_pred]

    if mask.shape[-2:] != pred_c.shape[-2:]:
        m_flat = mask.flatten(0, 1)  # (B*T,1,Hm,Wm)
        m_flat = F.interpolate(m_flat, size=pred_c.shape[-2:], mode="nearest")
        mask = m_flat.unflatten(0, (pred_c.size(0), T_pred))  # (B,T,1,H,W)

    diff = diff * mask
    denom = mask.sum() * C + 1e-8
    return diff.sum() / denom


def ensure_3ch_input(X_in, m):
    """
    Делает из X_in (B,T,2,H,W) вход с 3 каналами (u,v,mask):
    - если m=None -> ВОЗВРАЩАЕМ None (не подменяем фейковой маской из единиц)
    - если m есть, приводим её к (B,T,1,H,W) и подгоняем по времени/пространству
    """
    B, T, C, H, W = X_in.shape
    if C == 3:
        return X_in

    if m is None:
        return None  # важно: не искажаем вход

    M = m
    if M.ndim == 4:  # (B,1,H,W) -> (B,1,1,H,W)
        M = M.unsqueeze(2)
    if M.ndim == 5 and M.size(2) != 1:
        M = M[:, :, :1]

    # по времени -> T
    if M.size(1) > T:
        M = M[:, :T]
    elif M.size(1) < T:
        if M.size(1) == 1:
            M = M.repeat(1, T, 1, 1, 1)
        else:
            k = (T + M.size(1) - 1) // M.size(1)
            M = M.repeat(1, k, 1, 1, 1)[:, :T]

    # по пространству -> (H,W)
    if M.shape[-2:] != (H, W):
        M_flat = M.flatten(0, 1)  # (B*T,1,Hm,Wm)
        M_flat = F.interpolate(M_flat, size=(H, W), mode="nearest")
        M = M_flat.unflatten(0, (X_in.size(0), T))  # (B,T,1,H,W)

    return torch.cat([X_in, M.float()], dim=2)


# sequence length
seq_len = 10
pred_len = 10
total_len = seq_len + pred_len
Start_fatigue_cycles = 3000
cycle_gap = 18
SEED = 233

# MODE
MODE = "Crack"
bs = 2

If_Update_X = False
calculate_u_all = True

if MODE == "MovingMNIST":
    MODE_FOLDER = "MovingMNIST__seq_len10pred_len10__SimVP_Model_l0.0237913__20230926_140449"
    MODE_NAME = MODE_FOLDER + ".pt"
    EXPERIMENT = "MovingMNIST"
    model = nets.SimVP_Model(
        in_shape=[seq_len, 3, 128, 128],
        hid_S=64,
        hid_T=512,
        N_S=4,
        N_T=8,
        model_type="gSTA",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=True,
    )
    print("Decoder[0] conv stack:", model.dec.dec[0].conv)
    LOAD_PATH = os.path.join(
        "data",
        "seq_len" + str(seq_len) + "pred_len" + str(pred_len),
        "train_val_test_data_each_condition_pkl_3",
        EXPERIMENT,
        "train_val_test_data",
    )
    TEST_FIGURES_SAVE_PATH = [
        os.path.join(
            "res",
            "OutputImagesAndModelsForResults",
            "test_figures",
            MODE_FOLDER,
            "_" + "Update_X_" + str(If_Update_X),
            "0",
        )
    ]
elif MODE == "Crack":
    Discussion = "Discussion1"
    Model_name = "SimVP1"
    Condition = "FS_C_0_20"

    model = nets.SimVP_Model(
        in_shape=[seq_len, 3, 128, 128],  # 3 канала на вход
        hid_S=64,
        hid_T=512,
        N_S=4,
        N_T=8,
        model_type="gSTA",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=True,
    )

    MODEL_PATH = os.path.join("res", "OutputImagesAndModelsForResults", "models")
    MODE_FOLDER = Load_time_index(Discussion, Condition, MODEL_PATH)
    MODE_NAME = MODE_FOLDER + ".pt"

    ConditionLoadData = "FS_C_0_20"
    LOAD_PATH = os.path.join(
        "data",
        "seq_len" + str(seq_len) + "pred_len" + str(pred_len),
        "train_val_test_data_each_condition_or_discussion_pkl_3",
        Discussion,
        ConditionLoadData,
        "train_val_test_data",
    )
    TEST_FIGURES_SAVE_PATH = [
        os.path.join(
            "res",
            "OutputImagesAndModelsForResults",
            "test_figures",
            MODE_FOLDER,
            Discussion + "_" + ConditionLoadData + "_" + "Update_X_" + str(If_Update_X),
            "0",
        )
    ]
    TEST_SSIM_MSE_SAVE_PATH = os.path.join(
        "res",
        "OutputImagesAndModelsForResults",
        "test_figures",
        MODE_FOLDER,
        Discussion + "_" + ConditionLoadData + "_" + "Update_X_" + str(If_Update_X),
    )

starting_time = time.time()

for p in TEST_FIGURES_SAVE_PATH:
    os.makedirs(p, exist_ok=True)
os.makedirs(TEST_SSIM_MSE_SAVE_PATH, exist_ok=True)

with open(os.path.join(LOAD_PATH, "data_test.pkl"), "rb") as file:
    deserialized_data_test = pickle.load(file)

# dataset / loader
datasets = {"test": Datasets(data=deserialized_data_test, seq_len=seq_len, pred_len=pred_len)}
dataloaders = {
    "test": DataLoader(
        datasets["test"],
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
}

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch.cuda.is_available()", torch.cuda.is_available())
MODE_PATH = os.path.join("res", "OutputImagesAndModelsForResults", "models", MODE_FOLDER, MODE_NAME)
print("MODE_PATH:", MODE_PATH)
print("Checkpoint mtime:", time.ctime(os.path.getmtime(MODE_PATH)))
model.load_state_dict(torch.load(MODE_PATH, map_location="cpu"))
model = model.to(device)
model.eval()
print(f"✅ Model on device: {next(model.parameters()).device}")

record_and_plot_test_SSIM = record_and_plot_test_SSIM(save_path=TEST_SSIM_MSE_SAVE_PATH)
record_and_plot_test_MSE = record_and_plot_test_MSE(save_path=TEST_SSIM_MSE_SAVE_PATH)

# ============================ TEST ================================
outs = []
with torch.no_grad():
    for index, batch in enumerate(dataloaders["test"]):
        x, y, m = unpack_batch(batch)

        # вход в модель
        if index == 0 or (index != 0 and If_Update_X is False):
            X_in = x
        else:
            X_in = transforms.Update_X_with_out(x, outputs, pred_len, index)
            # важная страховка: не теряем маску (3-й канал) при обновлении входа
            if X_in.size(2) == 2 and x.size(2) >= 3:
                X_in = torch.cat([X_in, x[:, :, 2:3]], dim=2)

        X_in = ensure_3ch_input(X_in, m)
        if X_in is None:
            print(f"[SKIP@{index}] no mask channel in input — skip")
            continue

        X, Y = X_in.to(device), y.to(device)
        out = model(X)

        # лосс считаем без маски (она строится ниже, это инференс)
        y_loss = Y[:, :, :out.size(2), ...]
        loss = masked_mse_loss(out, y_loss, None)

        # ----- маска для визуализации: из ПОСЛЕДНЕГО входного кадра -----
        mask = None
        if X.ndim == 5 and X.size(2) >= 3:
            mask_in = X[:, seq_len - 1 : seq_len, 2:3]  # [B,1,1,H,W]
            mask = (mask_in.repeat(1, pred_len, 1, 1, 1) > 0.5).to(device)

        # накопление выходов (если нужно обновлять X)
        if If_Update_X is True:
            outs.append(out.cpu())
            outputs = torch.cat(outs, dim=1)

        if calculate_u_all is True:
            # |u| = sqrt(u^2 + v^2)
            Y_all = torch.sqrt(Y[:, :, 0, :, :] ** 2 + Y[:, :, 1, :, :] ** 2)  # [B,T,H,W]
            Out_all = torch.sqrt(out[:, :, 0, :, :] ** 2 + out[:, :, 1, :, :] ** 2)

            # ------------------- SSIM/MSE расчёт -------------------
            # применяем тот же crop ко всем данным
            # --- применяем тот же crop ко всем данным
            BORD = compute_border(Y_all.shape[-1], percent=0.1, min_px=4)
            Y_eval = crop_border_nd(Y_all, BORD)
            out_eval = crop_border_nd(Out_all, BORD)

            # --- добавляем ось канала ---
            if Y_eval.ndim == 4:
                Y_eval = Y_eval.unsqueeze(2)
            if out_eval.ndim == 4:
                out_eval = out_eval.unsqueeze(2)

            # --- обрезаем mask по тем же границам! ---
            if mask is not None:
                mask = crop_border_nd(mask.float(), BORD)
                if mask.ndim == 4:
                    mask = mask.unsqueeze(2)
                mask = mask[:, :, :1, :, :] > 0.5
            else:
                mask = None

            # --- теперь всё одной формы ---
            finite = torch.isfinite(Y_eval) & torch.isfinite(out_eval)
            valid_mask = finite if mask is None else (mask & finite)

            # --- гарантируем 5-мерный формат ---
            if valid_mask.ndim == 4:
                valid_mask = valid_mask.unsqueeze(2)
            valid_mask = valid_mask[:, :, :1, :, :]

            vm_y   = valid_mask.expand_as(Y_eval)
            vm_out = valid_mask.expand_as(out_eval)

            Y_eval_m   = Y_eval.masked_fill(~vm_y, 0.0)
            out_eval_m = out_eval.masked_fill(~vm_out, 0.0)




            print("Y_eval:", Y_eval.shape)
            print("valid_mask:", valid_mask.shape)




            # устойчивые ошибки (динамическое eps) и клип ТОЛЬКО для визуализации
            eps_dyn = max(1e-6, 0.01 * torch.nanmean(torch.abs(Y_all)).item())
            Absolute_error = torch.abs(Out_all - Y_all)
            denom = torch.clamp(Y_all.abs(), min=eps_dyn)
            rel_err_u = Absolute_error / denom
            rel_err_u = rel_err_u.masked_fill(Y_all.abs() < eps_dyn, float("nan"))
            try:
                rel_cap = torch.nanquantile(rel_err_u, 0.995).item()
                if not (np.isfinite(rel_cap) and rel_cap > 0):
                    rel_cap = 5.0
            except Exception:
                rel_cap = 5.0
            Relative_error = torch.clamp(rel_err_u, max=rel_cap)


            Plot_test_figures = Plot_test_figures_Chinese_u_all
            Y_plot, Out_plot = Y_all, Out_all

        else:
            # стандартная ветка u,v
            BORD = compute_border(out.shape[-1], percent=0.1, min_px=4)
            Y = crop_border_nd(Y, BORD)     # [B,T,Cy,Hc,Wc]
            out = crop_border_nd(out, BORD) # [B,T,Co,Hc,Wc]

            # --- пересечение валидности ---
            finite_common = torch.isfinite(Y).all(dim=2, keepdim=True) & torch.isfinite(out).all(dim=2, keepdim=True)  # [B,T,1,Hc,Wc]

            valid_mask = finite_common  # базово — только пересечение финитности
            if mask is not None:
                m = crop_border_nd(mask.float(), BORD) > 0.5                  # [B,T,1,Hc,Wc]
                valid_mask = valid_mask & m

            # визуализация: NaN вне mask-пересечения
            mfull_y  = valid_mask.expand(-1, -1, Y.size(2),   -1, -1)
            mfull_out= valid_mask.expand(-1, -1, out.size(2), -1, -1)
            Y_vis  = Y.masked_fill(~mfull_y,   float("nan"))
            Out_vis= out.masked_fill(~mfull_out, float("nan"))

            vr = valid_mask.float().mean().item()
            if vr < 0.01:
                print(f"[SKIP@{index}] too small valid area (ratio={vr:.4f})")
                continue

            # ошибки строго по пересечению валидных пикселей
            Absolute_error, Relative_error = evaluate.Calculate_Absolute_error_and_Relative_error(
                out, Y, valid_mask=valid_mask
            )
            Plot_test_figures = Plot_test_figures_Chinese
            Y_plot, Out_plot = Y_vis, Out_vis


        # ---------- сохраняем картинки ТОЛЬКО для выбранных индексов ----------
        if index in SAVE_ONLY_INDEXES:
            Plot_test_figures(
                X.cpu().numpy(),
                Y_plot.cpu().numpy(),
                Out_plot.cpu().numpy(),
                Absolute_error.cpu().numpy(),
                Relative_error.cpu().numpy(),
                index,
                TEST_FIGURES_SAVE_PATH,
                seq_len=seq_len,
                Start_fatigue_cycles=Start_fatigue_cycles,
                cycle_gap=cycle_gap,
            )


        # 🔄 Пересоздаём mask заново из входа X (в полном размере 128×128)
        mask = None
        if X.ndim == 5 and X.size(2) >= 3:
            mask_in = X[:, seq_len - 1 : seq_len, 2:3]  # [B,1,1,H,W]
            mask = (mask_in.repeat(1, pred_len, 1, 1, 1) > 0.5).to(device)

        # ---------- метрики ----------
        # ---------- метрики ----------
        C_eval = min(Y.size(2), out.size(2))
        Y_eval = Y[:, :, :C_eval, ...]
        out_eval = out[:, :, :C_eval, ...]

        finite = torch.isfinite(Y_eval) & torch.isfinite(out_eval)

        if mask is not None:
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            mask = mask[:, :, :1, :, :] > 0.5
            valid_mask = finite & mask
        else:
            valid_mask = finite

        if valid_mask.ndim == 4:
            valid_mask = valid_mask.unsqueeze(2)
        valid_mask = valid_mask[:, :, :1, :, :]

        vm_y   = valid_mask.expand_as(Y_eval)
        vm_out = valid_mask.expand_as(out_eval)

        Y_eval_m   = Y_eval.masked_fill(~vm_y, 0.0)
        out_eval_m = out_eval.masked_fill(~vm_out, 0.0)











        # ==========================================================
        # Crack growth length comparison plot
        # ==========================================================
        if 'Y_all' in locals() and 'Out_all' in locals():
            try:
                from tools.plot_crack_growth import plot_crack_growth
                from tools.overlay_crack_on_field import overlay_crack_on_field
                from tools.export_crack_metrics_tool import export_metrics_csv

                # optional:
                # from tools.publication_panels import publication_panels
                import numpy as np
                import os

                gt_disps, pred_disps = {}, {}
                gt_coords, gt_masks = {}, {}

                B, T, C, H, W = Y_all.shape  # C может быть 1 (u) или 2 (u,v)
                for b in range(B):
                    for t in range(T):
                        idx = b*T + t

                        # support C==1 or C>=2
                        if C >= 2:
                            # store as tuple (u,v) — это удобно дальше
                            gt_disps[idx]   = (Y_all[b, t, 0].cpu().numpy(),
                                            Y_all[b, t, 1].cpu().numpy())
                            pred_disps[idx] = (Out_all[b, t, 0].cpu().numpy(),
                                            Out_all[b, t, 1].cpu().numpy())
                        else:
                            gt_disps[idx]   = Y_all[b, t, 0].cpu().numpy()       # [H,W]
                            pred_disps[idx] = Out_all[b, t, 0].cpu().numpy()     # [H,W]

                        # (Y,X) grid to match your crack_length_mm usage
                        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                        gt_coords[idx] = (yy, xx)

                        if valid_mask is not None:
                            m = valid_mask[b, t]
                            if m.ndim == 3:  # [1,H,W]
                                m = m[0]
                            gt_masks[idx] = (m.detach().cpu().numpy() > 0.5).astype(np.uint8)
                        else:
                            gt_masks[idx] = np.ones((H, W), dtype=np.uint8)

                pred_coords = gt_coords
                pred_masks  = gt_masks

                save_dir = os.path.join(SAVE_PATH, "CrackGrowth")

                print(">>> Running plot_crack_growth")
                plot_crack_growth(gt_disps, gt_coords, gt_masks,
                                pred_disps, pred_coords, pred_masks, save_dir)

                print(">>> Running overlay_crack_on_field")
                overlay_crack_on_field(gt_disps, gt_coords, gt_masks,
                                    pred_disps, pred_coords, pred_masks, save_dir, which="pred")

                print(">>> Running export_metrics_csv")
                export_metrics_csv(gt_disps, gt_coords, gt_masks,
                                pred_disps, pred_coords, pred_masks, save_dir)

                # optional:
                # print(">>> Running publication_panels")
                # publication_panels(gt_disps, gt_coords, gt_masks,
                #                    pred_disps, pred_coords, pred_masks, save_dir)

            except Exception as e:
                print(f"⚠️ Post-processing failed: {e}")

            # ВНЕ try: чтобы были доступны дальше
            pred_masks  = gt_masks
            pred_coords = gt_coords
            save_dir    = os.path.join(SAVE_PATH, "CrackGrowth")

            # # Вызов с 6 аргументами и правильными coords
            # plot_crack_growth(
            #     gt_disps, gt_coords, gt_masks,
            #     pred_disps, pred_coords, pred_masks, save_dir
            # )
            

            try:
                print(">>> Running plot_crack_growth")
                plot_crack_growth(gt_disps, gt_coords, gt_masks,
                                pred_disps, pred_coords, pred_masks, save_dir)
            except Exception as e:
                print(f"⚠️ Crack growth plot failed: {e}")

            try:
                print(">>> Running overlay_crack_on_field")
                overlay_crack_on_field(gt_disps, gt_coords, gt_masks,
                                    pred_disps, pred_coords, pred_masks, save_dir)
            except Exception as e:
                print(f"⚠️ Overlay failed: {e}")

            try:
                print(">>> Running export_metrics_csv")
                export_metrics_csv(gt_disps, gt_coords, gt_masks,
                                pred_disps, pred_coords, pred_masks, save_dir)
            except Exception as e:
                print(f"⚠️ CSV export failed: {e}")

            try:
                print(">>> Running publication_panels")
                publication_panels(gt_disps, gt_coords, gt_masks,
                                pred_disps, pred_coords, pred_masks, save_dir)
            except Exception as e:
                print(f"⚠️ Publication panel failed: {e}")





                # save_dir = os.path.join(SAVE_PATH, "CrackGrowth")
                # plot_crack_growth(gt_disps, gt_coords, gt_masks, pred_disps, gt_coords, pred_masks, save_dir)




                # except Exception as e:
                #     print(f"⚠️ Crack growth plot failed: {e}")


        # ==========================================================
        # Crack skeleton overlay visualization
        # ==========================================================
        try:
            from src.utils.crack import detect_crack_skeleton
            from src.utils.plot import overlay_crack_on_field

            print("🎨 Drawing crack overlays on |u| fields...")

            save_overlay_dir = os.path.join(SAVE_PATH, "CrackOverlays")
            os.makedirs(save_overlay_dir, exist_ok=True)

            for i in range(min(len(gt_disps), 5)):  # максимум 5 кадров для примера
                gt_u, gt_v = gt_disps[i][0], gt_disps[i][1]
                pr_u, pr_v = pred_disps[i][0], pred_disps[i][1]
                mask = gt_masks[i] if gt_masks else None

                sk_gt = detect_crack_skeleton(gt_u, gt_v, valid_mask=mask)
                sk_pr = detect_crack_skeleton(pr_u, pr_v, valid_mask=mask)

                overlay_crack_on_field(
                    gt_u, gt_v,
                    skeleton_gt=sk_gt, skeleton_pred=sk_pr,
                    save_path=save_overlay_dir,
                    title=f"frame_{i:03d}"
                )

        except Exception as e:
            print(f"⚠️ Crack overlay drawing failed: {e}")


        # ==========================================================
        # Export crack length + SSIM/MSE metrics to CSV
        # ==========================================================
        try:
            from tools.export_crack_metrics_tool import export_metrics_csv


            print("🧾 Exporting crack metrics to CSV...")
            # Собираем SSIM/MSE, если они есть
            ssim_vals = getattr(record_and_plot_test_SSIM, "all_SSIM", None)
            mse_vals  = getattr(record_and_plot_test_MSE, "all_MSE", None)

            csv_dir = os.path.join(SAVE_PATH, "CrackGrowth")
            export_metrics_csv(
                save_dir=csv_dir,
                gt_disps=gt_disps,
                gt_coords=gt_coords,
                gt_masks=gt_masks,
                pred_disps=pred_disps,
                pred_coords=gt_coords,
                pred_masks=pred_masks,
                ssim_vals=ssim_vals,
                mse_vals=mse_vals,
            )

        except Exception as e:
            print(f"⚠️ CSV export failed: {e}")








        # ==========================================================
        # Publication-style comparison panel
        # ==========================================================
        try:
            from src.utils.crack import detect_crack_skeleton
            from src.utils.plot import publication_panel

            print("🖼️  Building publication panels for first few frames...")

            pub_dir = os.path.join(SAVE_PATH, "PublicationPanels")
            os.makedirs(pub_dir, exist_ok=True)

            for i in range(min(len(gt_disps), 3)):  # 3 примера достаточно
                gt_u, gt_v = gt_disps[i][0], gt_disps[i][1]
                pr_u, pr_v = pred_disps[i][0], pred_disps[i][1]
                mask = gt_masks[i] if gt_masks else None

                sk_gt = detect_crack_skeleton(gt_u, gt_v, valid_mask=mask)
                sk_pr = detect_crack_skeleton(pr_u, pr_v, valid_mask=mask)

                publication_panel(
                    gt_u, gt_v, pr_u, pr_v,
                    mask=mask,
                    skeleton_gt=sk_gt, skeleton_pred=sk_pr,
                    ssim_val=float(record_and_plot_test_SSIM.mean_SSIM()) if 'record_and_plot_test_SSIM' in locals() else None,
                    mse_val=float(record_and_plot_test_MSE.mean_MSE()) if 'record_and_plot_test_MSE' in locals() else None,
                    save_path=pub_dir,
                    title=f"Frame_{i:03d}"
                )
        except Exception as e:
            print(f"⚠️ Publication panel failed: {e}")

            


        ssim = evaluate.get_ssim(Y_eval_m.cpu().numpy(), out_eval_m.cpu().numpy())
        mse  = evaluate.get_mse (Y_eval_m.cpu(),        out_eval_m.cpu())
        record_and_plot_test_SSIM.record_SSIM(ssim)
        record_and_plot_test_MSE.record_MSE(mse)


# финальные графики метрик / сохранение
record_and_plot_test_SSIM.Calculate_Mean_SSIM(MODE_FOLDER, Discussion, ConditionLoadData)
record_and_plot_test_MSE.Calculate_Mean_MSE(MODE_FOLDER, Discussion, ConditionLoadData)
record_and_plot_test_SSIM.plot_test_SSIM(seq_len=seq_len, Start_fatigue_cycles=Start_fatigue_cycles, cycle_gap=cycle_gap)
record_and_plot_test_SSIM.save_SSIM(seq_len=seq_len, Start_fatigue_cycles=Start_fatigue_cycles, cycle_gap=cycle_gap, Model_name=Model_name)
record_and_plot_test_MSE.save_RMSE(seq_len=seq_len, Start_fatigue_cycles=Start_fatigue_cycles, cycle_gap=cycle_gap, Model_name=Model_name)
record_and_plot_test_MSE.plot_test_MSE(seq_len=seq_len, Start_fatigue_cycles=Start_fatigue_cycles, cycle_gap=cycle_gap)

end_time = (time.time() - starting_time) / 60
print(f"\nTesting Time: {end_time:.0f} min")
print("end!")
