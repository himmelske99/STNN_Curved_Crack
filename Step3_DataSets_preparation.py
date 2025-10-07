import os
import sys
import pickle
import torch
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# ---- PARAMS ----
seq_len = 10
pred_len = 10
# seq_len = 4
# pred_len = 4
Discussion = 'Discussion1'
Condition  = 'FS_C_0_20'

TRAIN_EXPERIMENT = ['FS_C_0_20_3']
VAL_EXPERIMENT   = ['FS_C_0_20_2']
TEST_EXPERIMENT  = ['FS_C_0_20_1']

SAVE_PATH = os.path.join(
    '..', 'data', f'seq_len{seq_len}pred_len{pred_len}',
    'train_val_test_data_each_condition_or_discussion_pkl_3',
    Discussion, Condition, 'train_val_test_data'
)
os.makedirs(SAVE_PATH, exist_ok=True)


def _specimen_pkl_path(exp: str, stage_type: str) -> str:
    """
    Ищет файл data_<stage_type>.pkl в любой подпапке train_val_test_data_each_specimen_pkl_*
    и возвращает первый найденный путь.
    """
    base = os.path.join('..', 'data', f'seq_len{seq_len}pred_len{pred_len}')
    candidates = []
    for name in os.listdir(base):
        if name.startswith('train_val_test_data_each_specimen_pkl_'):
            p = os.path.join(base, name, Discussion, exp, f'data_{stage_type}.pkl')
            if os.path.exists(p):
                candidates.append(p)
    if not candidates:
        return ""
    # Берём самый «свежий» по имени (с большим числом на конце)
    return sorted(candidates, reverse=True)[0]


def merge_and_save(stage_type: str, experiment_list: list[str]) -> None:
    if not experiment_list:
        logger.warning(f"{stage_type}: список экспериментов пуст — пропуск.")
        return

    logger.info(f"🔄 Merging {stage_type.upper()} from: {experiment_list}")
    data_list = []
    base_shape = None

    for exp in experiment_list:
        pkl_path = _specimen_pkl_path(exp, stage_type)
        if not pkl_path:
            logger.warning(f"{stage_type}: файл для {exp} не найден — пропуск")
            continue

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"{stage_type}: не удалось прочитать {pkl_path}: {e} — пропуск {exp}")
            continue

        if not isinstance(data, torch.Tensor):
            logger.warning(f"{stage_type}: {exp} вернул {type(data)} вместо torch.Tensor — пропуск")
            continue
        if data.numel() == 0 or data.shape[0] == 0:
            logger.warning(f"{stage_type}: {exp} пустой тензор — пропуск")
            continue

        # Проверка формата [N, C, H, W]
        if base_shape is None:
            base_shape = tuple(data.shape[1:])
        if tuple(data.shape[1:]) != base_shape:
            logger.error(f"{stage_type}: несовпадение формы у {exp}: {tuple(data.shape)} vs base {base_shape} — пропуск")
            continue

        logger.info(f"✅ Loaded from {exp}: shape={tuple(data.shape)}")
        data_list.append(data)

    if not data_list:
        logger.warning(f"{stage_type}: нечего сливать — итог пуст. Пропуск сохранения.")
        return

    merged = torch.cat(data_list, dim=0)
    logger.info(f"📦 Final merged {stage_type} shape: {tuple(merged.shape)}")

    out_path = os.path.join(SAVE_PATH, f'data_{stage_type}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(merged, f)
    logger.info(f"#Merged {stage_type} data saved to {out_path}")


if __name__ == '__main__':
    print("\n📁 Merging PKL datasets across experiments by condition...\n")
    merge_and_save('train', TRAIN_EXPERIMENT)
    merge_and_save('val',   VAL_EXPERIMENT)
    merge_and_save('test',  TEST_EXPERIMENT)
    print("\n🏁 Done.")
