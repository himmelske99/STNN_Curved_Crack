print(">>> LOADED FILE:", __file__, flush=True)


from multiprocessing import Pool, cpu_count
import os
import sys
import time
import pickle
import torch
from loguru import logger
logger.remove()
logger.add(sys.stdout, level='INFO')

sys.path.append("..")
from src.dataprocessing.dataset import load_data, data_after_transform_and_tofivedimension
from src.dataprocessing.transforms import get_traindata_mean_std

def _has_step1_data(split, stage):
    root = os.path.join('..', 'data', 'data_raw_pt_1', 'ValidData', split, stage)
    return os.path.exists(os.path.join(root, 'InputData.pt'))

# PARAMETERS
Discussion = 'Discussion1'
# seq_len = 4
# pred_len = 4

seq_len = 10
pred_len = 10
Need_plot = True

# Experiment structure

def list_stages_from_step1(split):
    base = os.path.join('..', 'data', 'data_raw_pt_1', 'ValidData', split)
    if not os.path.isdir(base):
        return []
    stages = []
    for name in os.listdir(base):
        if os.path.exists(os.path.join(base, name, 'InputData.pt')):
            stages.append(name)
    return sorted(stages)

TRAIN_EXPERIMENT_STAGE = list_stages_from_step1('FS_C_0_20_3')
VAL_EXPERIMENT_STAGE   = list_stages_from_step1('FS_C_0_20_2')
TEST_EXPERIMENT_STAGE  = list_stages_from_step1('FS_C_0_20_1')

print("🧠 Train (auto):", TRAIN_EXPERIMENT_STAGE)
print("🧪 Val   (auto):", VAL_EXPERIMENT_STAGE)
print("🧬 Test  (auto):", TEST_EXPERIMENT_STAGE)


def parallel_process_stage(split, stage, transform, seq_len, pred_len, need_plot):
    stage_path = os.path.join('..', 'data', 'data_raw_pt_1', 'ValidData', split, stage, 'InputData.pt')
    figure_path = os.path.join('..', 'res', 'OutputImagesForTestingCode',
                               'Step2_Data_preparation_figures', Discussion, split, stage)
    try:
        data = load_data(stage_path)
        if data is None:
            print(f"⚠️ {split}/{stage}: load_data вернул None — пропуск")
            return None

        data = data_after_transform_and_tofivedimension(
            data=data,
            transform=transform,
            seq_len=seq_len,
            pred_len=pred_len,
            figure_path=figure_path,
            Need_plot=need_plot
        )

        # import torch

        if data is None:
            print(f"⚠️ {split}/{stage}: после трансформов пусто — пропуск")
            return None

        # 1) Строим маску ИЗ ЭТОГО ЖЕ тензора (после всех аугментаций) и убираем NaN из u,v
        if data.ndim == 5:                        # [B, T, 2, H, W]
            mask = torch.isfinite(data).all(dim=2, keepdim=True)   # [B,T,1,H,W]
            data = torch.nan_to_num(data, nan=0.0)                 # NaN -> 0 в входах
            data = torch.cat([data, mask.float()], dim=2)          # -> [B,T,3,H,W]
        elif data.ndim == 4:                      # [N, 2, H, W]
            mask = torch.isfinite(data).all(dim=1, keepdim=True)   # [N,1,H,W]
            data = torch.nan_to_num(data, nan=0.0)
            data = torch.cat([data, mask.float()], dim=1)          # -> [N,3,H,W]
        else:
            raise RuntimeError(f"Неожиданная форма data: {tuple(data.shape)}")

        print(f"   post-transform+mask shape: {tuple(data.shape)}")  # контрольный лог

        # 2) ВАЖНО: вернуть результат из воркера!
        return data


    except RuntimeError as e:
        # Специально глушим «stack expects a non-empty TensorList»
        if "stack expects a non-empty TensorList" in str(e):
            print(f"⚠️ {split}/{stage}: пустой батч (stack) — пропуск")
            return None
        raise
    except Exception as e:
        print(f"❌ {split}/{stage}: ошибка воркера: {e} — пропуск")
        return None

def process_split(stages, split, transform):
    print(f"➡️ process_split: split={split}, requested_stages={len(stages)}")
    print(f"⚙️  Processing {split} data...")

    SAVE_PATH = os.path.join(
        '..', 'data',
        f'seq_len{seq_len}pred_len{pred_len}',
        f'train_val_test_data_each_specimen_pkl_{len(stages)}',
        Discussion, split
    )
    os.makedirs(SAVE_PATH, exist_ok=True)

    # filter stages that have Step1 outputs
    stages_available = [st for st in stages if _has_step1_data(split, st)]
    print(f"   ✔ available_stages={len(stages_available)}: {stages_available}")
    if not stages_available:
        print(f"⚠️ Нет данных шага 1 для {split}: ни одной стадии не найдено. Пропуск.")
        return

    # build worker args
    args = [(split, st, transform, seq_len, pred_len, Need_plot) for st in stages_available]

    from multiprocessing import Pool, cpu_count
    with Pool(min(max(cpu_count(), 1), 8)) as pool:
        results = pool.starmap(parallel_process_stage, args)

    # drop empty
    results = [r for r in results if r is not None]
    if not results:
        print(f"⚠️ Все стадии оказались пустыми для {split}. Пропуск записи датасета.")
        return

    import torch, pickle
    data_all = torch.cat(results, dim=0)
    print(f"{split}_data shape:", data_all.shape)

    split_map = {
        'FS_C_0_20_3': 'train',
        'FS_C_0_20_2': 'val',
        'FS_C_0_20_1': 'test'
    }
    save_name = split_map.get(split, split)
    out_pkl = os.path.join(SAVE_PATH, f'data_{save_name}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(data_all, f)
        logger.info(f"Saved: {out_pkl}")



if __name__ == "__main__":
    print(">>> MAIN START cwd:", os.getcwd(), flush=True)
    print("\n===== STEP 2: Data Preparation =====")
    print("🧠 Train:", TRAIN_EXPERIMENT_STAGE)
    print("🧪 Val:", VAL_EXPERIMENT_STAGE)
    print("🧬 Test:", TEST_EXPERIMENT_STAGE)
    start_time = time.time()

    train_transform = {'Resize': 128, 'Crop': 108, 'Rotation': 15, 'Flip': None}
    val_transform   = {'Resize': 128, 'Crop': 108, 'Rotation': 15, 'Flip': None}

    # test оставляем с SAFE_TEST_CROP_PERCENT как есть

    #################################
    RESIZE = 128
    SAFE_TEST_CROP_PERCENT = 0.9  # 96% от стороны после Resize (уменьшаем рамку)
    test_transform = {'Resize': RESIZE, 'Crop': int(round(RESIZE * SAFE_TEST_CROP_PERCENT))}
    ################################

    if TRAIN_EXPERIMENT_STAGE:
        print(">>> CALL process_split TRAIN", flush=True)
        process_split(TRAIN_EXPERIMENT_STAGE, 'FS_C_0_20_3', train_transform)
    if VAL_EXPERIMENT_STAGE:
        process_split(VAL_EXPERIMENT_STAGE, 'FS_C_0_20_2', val_transform)
    if TEST_EXPERIMENT_STAGE:
        process_split(TEST_EXPERIMENT_STAGE, 'FS_C_0_20_1', test_transform)

    elapsed = time.time() - start_time
    print(f"🏁 All done in {elapsed:.2f} seconds.")
    print("🚀 Multithreaded data preparation complete.")
    os.system('pause')
