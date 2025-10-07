# Step5_Check_mean_and_shape_of_data.py
import os
import sys
import pickle
import torch

sys.path.append("..")
from src.dataprocessing import transforms  # можно оставить: если там есть своя функция, попробуем её первой

# --- PARAMS ---
seq_len = 10
pred_len = 10
# seq_len = 4
# pred_len = 4
Discussion = 'Discussion1'
Condition  = 'FS_C_0_20'

LOAD_PATH = os.path.join(
    '..', 'data', f'seq_len{seq_len}pred_len{pred_len}',
    'train_val_test_data_each_condition_or_discussion_pkl_3',
    Discussion, Condition, 'train_val_test_data'
)
SAVE_PATH = os.path.join('..', 'OutputImagesForTestingCode', 'step5_Check_mean_and_shape_of_data')
os.makedirs(SAVE_PATH, exist_ok=True)

# --------- helpers: NAN-safe mean/std без torch.nanmean/nanstd ---------
def _masked_mean_std(x: torch.Tensor, reduce_dims):
    """
    NAN/INF-safe mean/std по каналам: mean = sum(x*mask)/count, var = E[x^2]-E[x]^2.
    Возвращает (mean, std), где размерность совпадает с x, сжата по reduce_dims.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.float()

    # маска валидных значений
    mask = torch.isfinite(x)  # True там, где не NaN/Inf
    # заменим невалидные значения нулями
    x0 = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))

    # считаем суммы/кол-во
    count = mask.sum(dim=reduce_dims)
    sum_  = x0.sum(dim=reduce_dims)
    sum2  = (x0 * x0).sum(dim=reduce_dims)

    # среднее
    mean = sum_ / count.clamp_min(1)
    mean = mean.masked_fill(count == 0, float('nan'))

    # дисперсия через E[x^2] - (E[x])^2
    ex2 = sum2 / count.clamp_min(1)
    var = torch.clamp(ex2 - mean * mean, min=0.0)
    var = var.masked_fill(count == 0, float('nan'))
    std = torch.sqrt(var)

    return mean, std

def _nan_safe_mean_std_any(x: torch.Tensor):
    """
    Возвращает (mean_c, std_c) по каналам C для 4D [N,C,H,W] и 5D [N,T,C,H,W].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 4:      # [N, C, H, W] -> сворачиваем по N,H,W
        reduce_dims = (0, 2, 3)
        mean_c, std_c = _masked_mean_std(x, reduce_dims)
    elif x.ndim == 5:    # [N, T, C, H, W] -> сворачиваем по N,T,H,W
        reduce_dims = (0, 1, 3, 4)
        mean_c, std_c = _masked_mean_std(x, reduce_dims)
    else:
        raise ValueError(f"Ожидается [N,C,H,W] или [N,T,C,H,W], получено {tuple(x.shape)}")
    return mean_c, std_c

def _valid_ratio_per_channel(x: torch.Tensor):
    """
    Доля валидных (не NaN/Inf) пикселей по каждому каналу C.
    Поддерживает 4D и 5D (см. выше).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 4:   # [N,C,H,W]
        total = x.shape[0]*x.shape[2]*x.shape[3]
        valid = torch.sum(torch.isfinite(x), dim=(0,2,3))
    elif x.ndim == 5: # [N,T,C,H,W]
        total = x.shape[0]*x.shape[1]*x.shape[3]*x.shape[4]
        valid = torch.sum(torch.isfinite(x), dim=(0,1,3,4))
    else:
        raise ValueError(f"Bad shape {tuple(x.shape)}")
    return (valid.float() / float(total)).cpu()

# --------------------- IO helpers ---------------------
def _load_pkl(path):
    if not os.path.exists(path):
        print(f"⚠️ Нет файла: {path}")
        return None
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception as e:
            print(f"❌ Не удалось прочитать {path}: {e}")
            return None

def _fmt_stats(name, data):
    if data is None:
        return f"-{name}: NO DATA\n"
    # сначала пробуем вашу функцию из transforms (если она уже nan-safe и поддерживает 5D)
    try:
        m, s = transforms.get_traindata_mean_std(data)
    except Exception:
        # fallback: наш универсальный NAN-safe вариант
        m, s = _nan_safe_mean_std_any(data)

    return (f"-shape of {name} dataset {tuple(data.shape)} "
            f"-mean X {float(m[0]):.7f} -std X {float(s[0]):.7f} "
            f"-mean Y {float(m[1]):.7f} -std Y {float(s[1]):.7f}\n")

# --------------------- main ---------------------
if __name__ == "__main__":
    train_pkl = os.path.join(LOAD_PATH, 'data_train.pkl')
    val_pkl   = os.path.join(LOAD_PATH, 'data_val.pkl')
    test_pkl  = os.path.join(LOAD_PATH, 'data_test.pkl')

    train = _load_pkl(train_pkl)
    val   = _load_pkl(val_pkl)
    test  = _load_pkl(test_pkl)

    # валидность каналов
    for name, arr in [("train", train), ("val", val), ("test", test)]:
        if arr is None:
            print(f"{name} valid ratio per channel: NO DATA")
        else:
            r = _valid_ratio_per_channel(arr)
            print(f"{name} valid ratio per channel:", [f"{float(x):.3f}" for x in r])

    # отчёт
    lines = []
    lines.append(f"-{Discussion}-{Condition}\n")
    lines.append(_fmt_stats("train", train))
    lines.append(_fmt_stats("val",   val))
    lines.append(_fmt_stats("test",  test))

    out_txt = os.path.join(SAVE_PATH, f"{Discussion}_{Condition}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("✅ Summary saved to:", out_txt)
    for line in lines:
        print(line, end="")
