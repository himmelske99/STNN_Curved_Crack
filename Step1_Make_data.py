# Step1_Make_data.py — FINAL
# Интерполяция displacement-полей с вырезанием пустот и сохранением масок
# Совместим с текущими src/ модулями и путями проекта

import os
import sys
sys.path.append("..")

import re
import time
import torch
from multiprocessing import Pool, cpu_count

from src.dataprocessing.datapreparation import import_data
from src.utils.utilityfunctions import (
    get_nodemaps_and_stage_nums,
    numpy_to_tensor,
    dict_to_list,
)
from src.dataprocessing.interpolation import interpolate_on_array
from src.utils.plot import interp_disp_images, origin_disp_images

import numpy as np
from scipy import ndimage as ndi
from src.dataprocessing.interpolation import _tight_bbox, _square_from_bbox




# ========================= Пользовательские параметры =========================
PIXELS = 128               # размер интерполяционной сетки (пикс)
VIS_EVERY = 1              # визуализировать каждую VIS_EVERY-ю стадию
CSV_EVERY = 1             # брать каждый n-й .csv внутри стадии
CSV_OFFSET = 0             # смещение выбора (0..CSV_EVERY-1)

# Режим окна интерполяции
ROI_MODE = "auto_stage"    # "auto_stage" | "fixed"
ROI_MARGIN = 0.00          # запас при auto_stage (0.05 = +5%)
FIXED_OFFSET = 0.0         # для режима "fixed": offset_x == offset_y == FIXED_OFFSET

# Параметры детекции «пустот»
MIN_PTS = 3                # минимум исходных точек в пикселе (2–5 обычно)
USE_HULL = True            # отсекать всё вне выпуклой оболочки
IDW_INSIDE = True          # мягкая дозаливка ТОЛЬКО внутри валидной маски
FILL_OUTSIDE = False       # ВАЖНО: не экстраполировать «за маской» (иначе дырки зальются)

# Поле зрения (мм) для каждой стадии (используется при ROI_MODE="fixed")
sizes = {
    'FS_C_0_20_1_A1':97, 'FS_C_0_20_1_B2':97, 'FS_C_0_20_1_C3':97, 'FS_C_0_20_1_D4':97, 'FS_C_0_20_1_E5':97,
    'FS_C_0_20_1_F6':97, 'FS_C_0_20_1_G7':97, 'FS_C_0_20_1_H8':97, 'FS_C_0_20_1_I9':97, 'FS_C_0_20_1_J10':97,
    'FS_C_0_20_2_A1':97, 'FS_C_0_20_2_B2':97, 'FS_C_0_20_2_C3':97, 'FS_C_0_20_2_D4':97, 'FS_C_0_20_2_E5':97,
    'FS_C_0_20_2_F6':97, 'FS_C_0_20_2_G7':97, 'FS_C_0_20_2_H8':97, 'FS_C_0_20_2_I9':97, 'FS_C_0_20_2_J10':97,
    'FS_C_0_20_3_A1':97, 'FS_C_0_20_3_B2':97, 'FS_C_0_20_3_C3':97, 'FS_C_0_20_3_D4':97, 'FS_C_0_20_3_E5':97,
    'FS_C_0_20_3_F6':97, 'FS_C_0_20_3_G7':97, 'FS_C_0_20_3_H8':97, 'FS_C_0_20_3_I9':97, 'FS_C_0_20_3_J10':97,
}

# Полный список образцов для прогонки (оставь как есть или сократи для теста)
EXPERIMENTS = {
    'FS_C_0_20_1': [
        'FS_C_0_20_1_A1', 'FS_C_0_20_1_B2', 'FS_C_0_20_1_C3', 'FS_C_0_20_1_D4', 'FS_C_0_20_1_E5',
        'FS_C_0_20_1_F6', 'FS_C_0_20_1_G7', 'FS_C_0_20_1_H8', 'FS_C_0_20_1_I9', 'FS_C_0_20_1_J10'
    ],
    'FS_C_0_20_2': [
        'FS_C_0_20_2_A1', 'FS_C_0_20_2_B2', 'FS_C_0_20_2_C3', 'FS_C_0_20_2_D4', 'FS_C_0_20_2_E5',
        'FS_C_0_20_2_F6', 'FS_C_0_20_2_G7', 'FS_C_0_20_2_H8', 'FS_C_0_20_2_I9', 'FS_C_0_20_2_J10'
    ],
    'FS_C_0_20_3': [
        'FS_C_0_20_3_A1', 'FS_C_0_20_3_B2', 'FS_C_0_20_3_C3', 'FS_C_0_20_3_D4', 'FS_C_0_20_3_E5',
        'FS_C_0_20_3_F6', 'FS_C_0_20_3_G7', 'FS_C_0_20_3_H8', 'FS_C_0_20_3_I9', 'FS_C_0_20_3_J10'
    ],
}

# ============================== Утилиты ======================================
def _natural_key(s: str):
    base = os.path.basename(str(s))
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', base)]

def _slice_every_n_list(lst, n: int, offset: int):
    lst_sorted = sorted(lst, key=_natural_key)
    return lst_sorted[offset::n]

def _slice_every_n_dict(dct, n: int, offset: int):
    items_sorted = sorted(dct.items(), key=lambda kv: _natural_key(kv[1]))
    return {
        k: v
        for i, (k, v) in enumerate(items_sorted)
        if i >= offset and (i - offset) % n == 0
    }

def _as_basenames(nodemaps):
    """Вернуть тот же тип (list/dict), но значения как basename (без директорий)."""
    if isinstance(nodemaps, dict):
        return {k: os.path.basename(v) for k, v in nodemaps.items()}
    return [os.path.basename(v) for v in nodemaps]

# ============================== Основная логика ===============================
def process_stage(args):
    experiment, stage_name, idx = args

    NODEMAP_PATH = os.path.join('..', 'data', 'data_raw_txt_0', 'ValidData', experiment, stage_name)
    SAVE_PATH_DATA = os.path.join('..', 'data', 'data_raw_pt_1', 'ValidData', experiment, stage_name)
    SAVE_PATH_IMAGE_ORIGIN = os.path.join('..', 'res', 'OutputImagesForTestingCode', 'Step1_Make_data_figures', experiment, stage_name, 'origin_disp')
    SAVE_PATH_IMAGE_INTERP = os.path.join('..', 'res', 'OutputImagesForTestingCode', 'Step1_Make_data_figures', experiment, stage_name, 'interp_disp')

    os.makedirs(SAVE_PATH_DATA, exist_ok=True)
    if idx % VIS_EVERY == 0:
        os.makedirs(SAVE_PATH_IMAGE_ORIGIN, exist_ok=True)
        os.makedirs(SAVE_PATH_IMAGE_INTERP, exist_ok=True)

    try:
        # 1) Список nodemap’ов стадии
        try:
            stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(NODEMAP_PATH)  # авто-парсинг шагов
        except TypeError:
            stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(NODEMAP_PATH, number=None)

        # Fallback: если нет словаря — взять просто *.csv
        if not stages_to_nodemaps:
            all_csv = [f for f in os.listdir(NODEMAP_PATH) if f.lower().endswith('.csv')]
            stages_to_nodemaps = sorted(all_csv, key=_natural_key)

        # 2) Разрежаем внутри стадии
        if isinstance(stages_to_nodemaps, dict):
            filtered_nodemaps = _slice_every_n_dict(stages_to_nodemaps, CSV_EVERY, CSV_OFFSET)
            num_before, num_after = len(stages_to_nodemaps), len(filtered_nodemaps)
        else:
            filtered_nodemaps = _slice_every_n_list(list(stages_to_nodemaps), CSV_EVERY, CSV_OFFSET)
            num_before, num_after = len(stages_to_nodemaps), len(filtered_nodemaps)

        if num_after == 0:
            print(f'⚠️ {experiment}/{stage_name}: после фильтрации нет файлов '
                  f'(CSV_EVERY={CSV_EVERY}, OFFSET={CSV_OFFSET}). Пропуск.')
            return

        # 3) Нормализуем к basename для import_data
        filtered_nodemaps = _as_basenames(filtered_nodemaps)
        example_name = next(iter(filtered_nodemaps.values())) if isinstance(filtered_nodemaps, dict) else filtered_nodemaps[0]
        print(f"→ Import {num_after}/{num_before} CSV @ {NODEMAP_PATH}; пример: {example_name}")

        # 4) Импорт только выбранных файлов
        inputs_disp = import_data(nodemaps=filtered_nodemaps, data_path=NODEMAP_PATH)

        # 5) Визуализация «сырья»
        if idx % VIS_EVERY == 0:
            origin_disp_images(inputs_disp, SAVE_PATH=SAVE_PATH_IMAGE_ORIGIN)

        # 6) Интерполяция (сплошное поле + авто-ROI по стадии) — ЕДИНСТВЕННЫЙ блок
        # 6) Интерполяция — ЕДИНСТВЕННЫЙ блок
        interp_kwargs = dict(
            pixels=PIXELS,
            roi_mode=ROI_MODE,
            roi_margin=ROI_MARGIN,

            # Вырезаем пустоты и НЕ экстраполируем за их пределы
            fill_outside=False,
            use_hull=True,

            # NEW: kNN-маска пустот (принципиально другой метод)
            adaptive_knn_mask=True,
            knn_k=12,
            knn_factor=1.6,

            # Мягкий докрас только внутри valid-маски
            idw_inside_mask=False,
            # idw_inside_mask=True,
        )

        interp_coords, interp_disps, interp_masks = interpolate_on_array(
            input_by_nodemap=inputs_disp,
            return_masks=True,
            **interp_kwargs
        )

        # ---- общий кроп по стадии ----
        # ---- АДАПТИВНЫЙ КВАДРАТНЫЙ КРОП ПО СТАДИИ ----
        # ---- УСТОЙЧИВЫЙ АДАПТИВНЫЙ КРОП ПО СТАДИИ ----
        min_coverage     = 0.92     # минимальная доля валидности пикселя во всех кадрах
        safety_px_init   = 2        # начальный отступ внутрь (эрозия)
        target_border_bad= 0.0      # допустимая доля "плохих" пикселей на периметре
        max_erosion_iter = 8        # максимум итераций усиления эрозии
        pad_before_square= 2        # лёгкий паддинг перед квадратизацией

        mask_list = [interp_masks[k].astype(bool) for k in interp_masks.keys()]
        mstack = np.stack(mask_list, axis=0)  # [N,H,W]
        H, W = mstack.shape[1:]
        vote = mstack.mean(axis=0)            # средняя валидность по кадрам [0..1]
        base = (vote >= min_coverage)

        # страховка, если base пустая
        if not np.any(base):
            print(f"⚠️  Base mask empty after coverage={min_coverage:.2f}, ослабляем критерий.")
            base = (vote > 0)  # хоть где-то валидные
            if not np.any(base):
                base[:] = True  # fallback: не режем вообще

        s = ndi.generate_binary_structure(2, 1)
        erosion = int(safety_px_init)
        best_bbox = None

        def _perimeter_bad_fraction(mstack_crop: np.ndarray) -> float:
            """Доля не полностью валидных пикселей на периметре."""
            mean_in_crop = mstack_crop.mean(axis=0)
            h, w = mean_in_crop.shape
            perim = np.ones((h, w), dtype=bool); perim[1:-1, 1:-1] = False
            bad = (mean_in_crop < 1.0) & perim
            return float(bad.mean())

        for i in range(max_erosion_iter + 1):
            region = ndi.binary_erosion(base, structure=s, iterations=max(erosion, 0)) if erosion > 0 else base
            if not np.any(region):
                if best_bbox is not None:
                    y0, y1, x0, x1 = best_bbox
                    print(f"⚠️  Erosion too strong (iter={i}), откат на последний валидный bbox.")
                    break
                region = vote > 0
            y0, y1, x0, x1 = _tight_bbox(region)
            y0, y1, x0, x1 = _square_from_bbox(y0, y1, x0, x1, H, W, pad=int(pad_before_square))
            mstack_crop = mstack[:, y0:y1, x0:x1]
            frac_bad = _perimeter_bad_fraction(mstack_crop)
            best_bbox = (y0, y1, x0, x1)

            if frac_bad <= target_border_bad + 1e-9:
                break
            erosion += 1

        # если bbox так и не найден (редкий случай) — fallback на весь кадр
        if best_bbox is None:
            print("⚠️  Crop bbox not found — using full frame.")
            y0, y1, x0, x1 = 0, H, 0, W
        else:
            y0, y1, x0, x1 = best_bbox

        print(f"   Auto-crop: y[{y0}:{y1}] x[{x0}:{x1}] side={y1-y0}px (erosion={erosion})")


        # применяем кроп ко всем данным стадии
        interp_coords_c, interp_disps_c, interp_masks_c = {}, {}, {}
        for k in interp_disps.keys():
            d = interp_disps[k][:, y0:y1, x0:x1]
            m = interp_masks[k][y0:y1, x0:x1]
            Xc = interp_coords[k][0][y0:y1, x0:x1]
            Yc = interp_coords[k][1][y0:y1, x0:x1]
            interp_disps_c[k]  = d
            interp_masks_c[k]  = m
            interp_coords_c[k] = np.array([Xc, Yc], dtype=np.float32)

        interp_coords = interp_coords_c
        interp_disps  = interp_disps_c
        interp_masks  = interp_masks_c

        # быстрый контроль: доля NaN внутри финального квадрата
        sample = next(iter(interp_disps.values()))
        u_map, v_map = sample[0], sample[1]
        nan_u = float(np.isnan(u_map).mean()); nan_v = float(np.isnan(v_map).mean())
        print(f"   Inside-crop NaN ratio: u={nan_u:.3f}, v={nan_v:.3f}")

        
        
        
        # # ---- общий кроп по стадии (один на все кадры) ----
        # MIN_COVERAGE = 0.90   # доля кадров, в которых пиксель валиден
        # SAFETY_PX    = 2      # «засунуться» внутрь на n пикселей (эрозия)
        # CROP_PAD     = 2      # лёгкий паддинг перед квадратизацией

        # # 1) стак масок -> [N,H,W]
        # mask_list = [interp_masks[k].astype(bool) for k in interp_masks.keys()]
        # mstack = np.stack(mask_list, axis=0)  # [N,H,W]
        # H, W = mstack.shape[1:]

        # # 2) голосование валидности
        # vote = mstack.mean(axis=0)            # [H,W] в [0..1]
        # base = (vote >= MIN_COVERAGE)

        # # 3) безопасный отступ от края домена
        # if SAFETY_PX > 0:
        #     s = ndi.generate_binary_structure(2, 1)
        #     base = ndi.binary_erosion(base, structure=s, iterations=int(SAFETY_PX))

        # # страховки, если вдруг пусто
        # if not base.any():
        #     base = (vote >= (MIN_COVERAGE * 0.8))
        #     if not base.any():
        #         base = (vote > 0)

        # # 4) tight-bbox -> квадрат
        # y0, y1, x0, x1 = _tight_bbox(base)
        # y0, y1, x0, x1 = _square_from_bbox(y0, y1, x0, x1, H, W, pad=int(CROP_PAD))
        # print(f"   Crop square: y[{y0}:{y1}] x[{x0}:{x1}] -> side={y1-y0} px")

        # # 5) применяем кроп ко ВСЕМ данным стадии
        # interp_coords_c, interp_disps_c, interp_masks_c = {}, {}, {}
        # for k in interp_disps.keys():
        #     # поля смещений [2,H,W]
        #     d = interp_disps[k][:, y0:y1, x0:x1]
        #     # маска [H,W]
        #     m = interp_masks[k][y0:y1, x0:x1]
        #     # координаты (X,Y) — те же срезы
        #     Xc = interp_coords[k][0][y0:y1, x0:x1]
        #     Yc = interp_coords[k][1][y0:y1, x0:x1]

        #     interp_disps_c[k]  = d
        #     interp_masks_c[k]  = m
        #     interp_coords_c[k] = np.array([Xc, Yc], dtype=np.float32)

        # # заменяем коллекции на подрезанные
        # interp_coords = interp_coords_c
        # interp_disps  = interp_disps_c
        # interp_masks  = interp_masks_c

        # # быстрый контроль NaN внутри квадрата
        # sample = next(iter(interp_disps.values()))
        # u_map, v_map = sample[0], sample[1]
        # nan_u = float(np.isnan(u_map).mean())
        # nan_v = float(np.isnan(v_map).mean())
        # print(f"   Inside-crop NaN ratio: u={nan_u:.3f}, v={nan_v:.3f}")




        # 7) Сохранение .pt (интерполированные карты после кропа)
        inputs = numpy_to_tensor(interp_disps, dtype=torch.float32)   # dict: {name: [1,2,Hc,Wc]}
        inputs_list = dict_to_list(inputs)
        torch.save(inputs_list, os.path.join(SAVE_PATH_DATA, 'InputData.pt'))

        # 8) Сохранение масок (тем же порядком)
        masks_np = {k: interp_masks[k].astype('uint8')[None, ...] for k in interp_masks}  # [1,Hc,Wc]
        masks_t  = dict_to_list({k: torch.tensor(v) for k, v in masks_np.items()})
        torch.save(masks_t, os.path.join(SAVE_PATH_DATA, 'MaskData.pt'))


        # 9) Быстрая проверка интерполяции (картинки u и v)
        if idx % VIS_EVERY == 0:
            interp_disp_images(inputs_list, SAVE_PATH=SAVE_PATH_IMAGE_INTERP,
                            component="u", crop_valid_bbox=True, crop_pad=2)
            interp_disp_images(inputs_list, SAVE_PATH=SAVE_PATH_IMAGE_INTERP,
                            component="v", crop_valid_bbox=True, crop_pad=2)


        
        # sanity-print: доля NaN по u и v (после интерполяции)
        first = next(iter(interp_disps.values()))
        u_map, v_map = first[0], first[1]
        nan_u = float((~torch.tensor(u_map, dtype=torch.float32).isfinite()).float().mean())
        nan_v = float((~torch.tensor(v_map, dtype=torch.float32).isfinite()).float().mean())
        print(f"   NaN ratio: u={nan_u:.3f}  v={nan_v:.3f}")
    except Exception as e:
        print(f"❌ Failed: {experiment}/{stage_name} — {e}")

# ============================== Запуск ========================================
if __name__ == "__main__":
    print("📂 Step-1: Interpolation with hole masking")
    print(f"Settings: PIXELS={PIXELS}, ROI_MODE={ROI_MODE}, ROI_MARGIN={ROI_MARGIN}, "
          f"MIN_PTS={MIN_PTS}, HULL={USE_HULL}, IDW_INSIDE={IDW_INSIDE}, FILL_OUTSIDE={FILL_OUTSIDE}")

    args = []
    for exp, stages in EXPERIMENTS.items():
        for i, st in enumerate(stages):
            args.append((exp, st, i))

    t0 = time.time()
    # Параллельно по стадиям (можно заменить на последовательный обход, если нужно)
    with Pool(min(cpu_count(), 8)) as pool:
        pool.map(process_stage, args)

    dt = time.time() - t0
    print(f"🏁 Done Step-1 in {dt:.1f}s")

