# ParallelNets_train.py
# coding:utf-8
import os
import pickle
from itertools import product
from datetime import datetime
from loguru import logger

import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.deep_learning.train import train_ConvLSTM
from src.docu.docu import Documentation
from src.dataprocessing import transforms
from src.dataprocessing.dataset import Datasets
from src.deep_learning import nets, evaluate


def unpack_batch(batch):
    """
    Принимает batch из DataLoader и возвращает (X, Y, mask_or_None)
    Поддерживает (X,Y) и (X,Y,mask).
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            X, Y, M = batch
            return X, Y, M
        elif len(batch) == 2:
            X, Y = batch
            return X, Y, None
    # на всякий случай — пытаемся разобрать стандартный случай
    try:
        X, Y = batch
        return X, Y, None
    except Exception:
        raise ValueError(f"Unexpected batch structure: {type(batch)} / len={getattr(batch,'__len__',lambda: '?')()}")

def masked_mse_loss(pred, target, mask=None, eps=1e-6):
    """
    Элементное MSE со средним по валидным пикселям.
    pred/target: [B,T,C,H,W] или совместимые.
    mask:        [B,T,1,H,W] (bool/0-1). Если None — обычный .mean().
    """
    diff2 = (pred - target) ** 2
    if mask is None:
        return diff2.mean()
    # приводим маску к форме diff2
    m = mask.bool()
    while m.ndim < diff2.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(diff2)
    valid_cnt = m.sum().clamp_min(1)
    return diff2[m].sum() / valid_cnt






# --- Paths ---
MODEL_PATH = os.path.join("res", "OutputImagesAndModelsForResults", "models")
EVALUATE_PATH = os.path.join("res", "OutputImagesAndModelsForResults", "evaluates")

# --- Training params ---
NUM_EPOCHS = 800
patience = 80

seq_len = 10
pred_len = 10
# seq_len = 4
# pred_len = 4

SEED = 19990329

# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class BestHyperparamDocu:
    """Store the best hyperparameters"""

    def __init__(self):
        self.i = -1
        self.mseloss = float("inf")
        self.lr = None
        self.dropout_prob = None
        self.bs = None
        self.sl = None
        self.mid_features = None
        self.num_hiddens = None
        self.output_features = None
        self.num_layers = None


def run(
    lr=1e-4,
    dropout_prob=0.0,
    bs=1,


    # sl=4,
    
    sl=10,
    mid_features=128,
    num_hiddens=256,
    output_features=2,
    num_layers=2,
    MODE="Crack",
):
    # --- Local lengths ---
    seq_len_local = sl
    pred_len_local = pred_len

    Hyperparameters = dict(
        lr=str(lr),
        drop=str(dropout_prob),
        bs=str(bs),
        sl=str(seq_len_local),
        mid_fea=str(mid_features),
        n_hidd=str(num_hiddens),
        out_fea=str(output_features),
        n_lay=str(num_layers),
    )
    print(
        f"- lr: {lr} - dropout: {dropout_prob} - bs: {bs} - mid: {mid_features} "
        f"- hidden: {num_hiddens} - layers: {num_layers} - seq_len: {seq_len_local}"
    )

    # --- Load data ---
    if MODE == "Crack":
        Discussion = "Discussion1"
        Condition = "FS_C_0_20"
        LOAD_PATH = os.path.join(
            "data",
            f"seq_len{seq_len}pred_len{pred_len}",
            "train_val_test_data_each_condition_or_discussion_pkl_3",
            Discussion,
            Condition,
            "train_val_test_data",
        )

        model = nets.SimVP_Model(
            in_shape=[seq_len_local, 3, 128, 128],   # именно 3 канала!
            hid_S=64, hid_T=512, N_S=4, N_T=8, model_type="gSTA",
            mlp_ratio=8.0, drop=0.0, drop_path=0.0,
            spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
        )


    with open(os.path.join(LOAD_PATH, "data_train.pkl"), "rb") as f:
        deserialized_data_train = pickle.load(f)
    with open(os.path.join(LOAD_PATH, "data_val.pkl"), "rb") as f:
        deserialized_data_val = pickle.load(f)
    with open(os.path.join(LOAD_PATH, "data_test.pkl"), "rb") as f:
        deserialized_data_test = pickle.load(f)

    # --- Mean / std ---
    data_train_mean, data_train_std = transforms.get_traindata_mean_std(deserialized_data_train)

    # --- Datasets & loaders ---
    datasets = {
        "train": Datasets(deserialized_data_train, seq_len=seq_len_local, pred_len=pred_len_local),
        "val": Datasets(deserialized_data_val, seq_len=seq_len_local, pred_len=pred_len_local),
        "test": Datasets(deserialized_data_test, seq_len=seq_len_local, pred_len=pred_len_local),
    }
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")

    dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, batch_size=bs,
                            num_workers=0, pin_memory=has_cuda, drop_last=True),
        "val": DataLoader(datasets["val"], shuffle=False, batch_size=bs,
                          num_workers=0, pin_memory=has_cuda, drop_last=False),
        "test": DataLoader(datasets["test"], shuffle=False, batch_size=1,
                           num_workers=0, pin_memory=has_cuda, drop_last=False),
    }
    for batch in dataloaders["train"]:
        X, Y, M = unpack_batch(batch)  # поддерживает и (X,Y), и (X,Y,mask)
        msg = f"BATCH SHAPES: X={tuple(X.shape)}, Y={tuple(Y.shape)}"
        if M is not None:
            msg += f", mask={tuple(M.shape)}"
        print(msg)
        break

    logger.info(f"#dataset_sizes: {dataset_sizes}")

    # --- Model to device ---
    model = model.to(device)

    # --- Criterion / Optimizer / Scheduler ---
    # --- Criterion / Optimizer / Scheduler ---
    # from src.deep_learning.loss import masked_mse_loss

    def criterion(pred, target, mask=None):
        # Универсальный masked MSE (nan-safe, mean по валидным пикселям)
        diff2 = (pred - target) ** 2
        if mask is not None:
            m = mask.bool()
            while m.ndim < diff2.ndim:
                m = m.unsqueeze(-1)
            m = m.expand_as(diff2)
            valid = m & torch.isfinite(diff2)
            return diff2[valid].mean()
        else:
            valid = torch.isfinite(diff2)
            return diff2[valid].mean()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(dataloaders["train"]))
    total_steps = NUM_EPOCHS * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, final_div_factor=1e4
    )


    # --- Evaluate setup ---
    if MODE == "MovingMNIST":
        name = f"{MODE}__seq{seq_len}pred{pred_len}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:  # Crack
        name = f"{Discussion}_{Condition}__seq{seq_len}pred{pred_len}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = os.path.join(EVALUATE_PATH, name)
    os.makedirs(path, exist_ok=True)

    evaluate_loss_accuracy_epoch = evaluate.evaluate_loss_accuracy_epoch(
        save_path=path, pred_len=pred_len_local
    )

    # --- Train ---
    model, train_docu = train_ConvLSTM(
        data_train_mean,
        data_train_std,
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        evaluate_loss_accuracy_epoch,
        Hyperparameters,
        scheduler,
        NUM_EPOCHS,
        device,
        seq_len=seq_len_local,
        pred_len=pred_len_local,
        patience=patience,
        grad_clip=1.0,   # enable gradient clipping
    )

    print(f"\nTraining Time: {train_docu.train_time:.0f} min")
    print(f"- Best MSELoss: {train_docu.mseloss:.7f}")

    # --- Save model ---
    if MODE == "MovingMNIST":
        name = f"{MODE}__seq{seq_len}pred{pred_len}__{model.__class__.__name__}_l{train_docu.mseloss:.7f}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        name = f"{Discussion}_{Condition}__seq{seq_len}pred{pred_len}__{model.__class__.__name__}_l{train_docu.mseloss:.7f}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    save_path = os.path.join(MODEL_PATH, name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, name + ".pt"))

    # --- Save metadata ---
    docu = Documentation(datasets, dataloaders, model, criterion, optimizer, scheduler, NUM_EPOCHS, train_docu)
    docu.save_metadata(path=save_path, name=name)

    return train_docu, Hyperparameters


if __name__ == "__main__":
    # Hyperparameters
    params = dict(
        lr=[0.0005],
        dropout_prob=[0.5],
        bs=[8],
        sl=[10],
        # sl=[4],
        mid_features=[128],
        num_hiddens=[512],
        output_features=[2],
        num_layers=[3],
    )

    NUM_RUNS = 1
    for run_id in range(NUM_RUNS):
        print(f"\nRun: {run_id + 1}/{NUM_RUNS}")

        best_docu = BestHyperparamDocu()
        i = 1
        for current_params in product(*list(params.values())):
            train_docu, Hyperparameters = run(*current_params)
            if train_docu.mseloss < best_docu.mseloss:
                best_docu.i = i + 1
                best_docu.mseloss = train_docu.mseloss
                best_docu.lr = Hyperparameters["lr"]
                best_docu.dropout_prob = Hyperparameters["drop"]
                best_docu.bs = Hyperparameters["bs"]
                best_docu.sl = Hyperparameters["sl"]
                best_docu.mid_features = Hyperparameters["mid_fea"]
                best_docu.num_hiddens = Hyperparameters["n_hidd"]
                best_docu.output_features = Hyperparameters["out_fea"]
                best_docu.num_layers = Hyperparameters["n_lay"]
            i += 1
        print(
            f"The best Hyperparameters are: "
            f"lr={best_docu.lr}, dropout={best_docu.dropout_prob}, bs={best_docu.bs}, "
            f"num_hiddens={best_docu.num_hiddens}, num_layers={best_docu.num_layers}"
        )
