"""P2 - FedIoV: tim sieu tham so KANConvNet bang Genetic Algorithm.

Bai bao dung DEAP. O day cai GA thuan numpy (tournament + uniform crossover +
mutation theo gene) de khong them phu thuoc; API giu tuong duong nen thay bang
DEAP rat de neu can.

Gene toi uu:
  lr, batch_size, dropout, width (c1,c2), grid_size, spline_order

Fitness = macro-F1 tren tap validation, do bang cach train tap trung NGAN
(--proxy-epochs) tren du lieu gop cua vai client. Day la "proxy task": re hon
chay ca vong FL cho moi ca the, dung cach bai bao mo ta.

Chay:
  python ga_search.py --pop 12 --generations 6 --clients 0 1 2 3
  python ga_search.py --pop 8 --generations 4 --proxy-epochs 2 --max-samples 50000
"""
import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402
from model_cnn1d import FocalLoss                # noqa: E402
from model_kanconv import KANConvNet, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Khong gian tim kiem: moi gene la chi so vao danh sach ben duoi
# Bang 3 cua bai (Heidari et al., FGCS 181 (2026) 108448) — chep NGUYEN VAN.
# Ban truoc dung gia tri tu nghi, khong mot con so nao trung voi bai.
SPACE = {
    "lr":            [0.00005, 0.0005, 0.005, 0.02, 0.05],      # alpha
    "batch_size":    [12, 24, 48, 96, 192],                     # beta
    "epochs":        [3, 7, 15, 30, 60],                        # e
    "momentum":      [0.2, 0.6, 0.85, 0.95, 0.998],             # mu
    "dropout":       [0.02, 0.08, 0.25, 0.45, 0.65],            # delta
    "optimizer":     ["AdamW", "RMSprop", "AdaDelta", "SGD", "Nadam"],   # omega
    "l2":            [0.00005, 0.0005, 0.005, 0.05],            # lambda
    "personal_coef": [0.15, 0.35, 0.55, 0.75, 0.95],            # alpha (local)
    # Bai KHONG cho width / grid_size trong Bang 3 -> giu lai nhu tham so cua ta
    "width":         [(8, 16), (16, 32), (24, 48), (32, 64)],
    "grid_size":     [3, 5, 8, 12],
}
# Pc, Pm cua Bang 3 la tham so CUA GA, khong phai gen -> dat o dong lenh
PC_VALUES = [0.65, 0.75, 0.85]
PM_VALUES = [0.02, 0.06, 0.12]
GENES = list(SPACE.keys())


def random_individual(rng):
    return {g: int(rng.integers(len(SPACE[g]))) for g in GENES}


def decode(ind):
    return {g: SPACE[g][ind[g]] for g in GENES}


def crossover(a, b, rng, p=0.5):
    return ({g: (a[g] if rng.random() < p else b[g]) for g in GENES},
            {g: (b[g] if rng.random() < p else a[g]) for g in GENES})


def mutate(ind, rng, rate):
    out = dict(ind)
    for g in GENES:
        if rng.random() < rate:
            out[g] = int(rng.integers(len(SPACE[g])))
    return out


def tournament(pop, fits, rng, k=3):
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    return dict(pop[max(idx, key=lambda i: fits[i])])


# ----------------------------------------------------------------------------
def evaluate_individual(ind, data, device, proxy_epochs, seed):
    """Train ngan roi tra ve macro-F1 tren validation."""
    hp = decode(ind)
    xtr, ytr, xva, yva = data
    torch.manual_seed(seed)

    model = KANConvNet(INPUT_LEN, NUM_GLOBAL_CLASSES, hp["dropout"],
                       hp["width"], hp["grid_size"], 3, "fourier").to(device)
    crit = FocalLoss(alpha=C.make_focal_alpha(ytr).to(device), gamma=2.0)
    ten_opt = hp.get("optimizer", "AdamW")
    l2 = hp.get("l2", 0.0)
    mo = hp.get("momentum", 0.9)
    if ten_opt == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=l2)
    elif ten_opt == "RMSprop":
        opt = optim.RMSprop(model.parameters(), lr=hp["lr"], momentum=mo,
                            weight_decay=l2)
    elif ten_opt == "AdaDelta":
        opt = optim.Adadelta(model.parameters(), lr=hp["lr"], weight_decay=l2)
    elif ten_opt == "SGD":
        opt = optim.SGD(model.parameters(), lr=hp["lr"], momentum=mo,
                        weight_decay=l2)
    else:                                    # Nadam
        opt = optim.NAdam(model.parameters(), lr=hp["lr"], weight_decay=l2)
    tr_loader = C.make_loader(xtr, ytr, hp["batch_size"], shuffle=True)
    va_loader = C.make_loader(xva, yva, 4096, shuffle=False)

    model.train()
    for _ in range(proxy_epochs):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    m, _, _ = C.evaluate(model, va_loader, crit, device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return m["macro_f1"], m, n_params


def main():
    p = argparse.ArgumentParser(description="P2 FedIoV - GA tim sieu tham so")
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--mutation-rate", type=float, default=0.25)
    p.add_argument("--proxy-epochs", type=int, default=2)
    p.add_argument("--clients", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--max-samples", type=int, default=60_000,
                   help="Tren MOI client, truoc khi gop")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(args.out_dir, "ga_search.log"))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- du lieu proxy ---
    xs, ys = [], []
    for cid in args.clients:
        xi, yi = C.load_client_data(args.data_dir, cid, args.task, args.max_samples)
        xs.append(xi)
        ys.append(yi)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    del xs, ys
    strat = y if np.bincount(y).min() >= 2 else None
    xtr, xva, ytr, yva = train_test_split(x, y, test_size=args.val_size,
                                          random_state=args.seed, stratify=strat)
    del x, y
    logger.info(f"Proxy task: train={len(ytr)}, val={len(yva)}, thiet bi={device}")
    data = (xtr, ytr, xva, yva)

    # --- GA ---
    csv_path = os.path.join(args.out_dir, "ga_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["generation", "individual", "macro_f1", "accuracy", "params",
             "seconds"] + GENES)

    pop = [random_individual(rng) for _ in range(args.pop)]
    best, best_fit = None, -1.0

    for gen in range(args.generations):
        fits = []
        for i, ind in enumerate(pop):
            t0 = time.time()
            try:
                fit, m, n_params = evaluate_individual(
                    ind, data, device, args.proxy_epochs, args.seed)
            except RuntimeError as e:                  # OOM voi cau hinh lon
                logger.warning(f"  ca the {i} loi ({e}); fitness=0")
                fit, m, n_params = 0.0, {"accuracy": 0.0}, 0
            dt = time.time() - t0
            fits.append(fit)
            hp = decode(ind)
            logger.info(f"[Gen {gen}] ca the {i}: macro_f1={fit:.4f} "
                        f"acc={m['accuracy']:.4f} params={n_params:,} "
                        f"({dt:.1f}s) {hp}")
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [gen, i, round(fit, 6), round(m["accuracy"], 6), n_params,
                     round(dt, 1)] + [hp[g] for g in GENES])
            if fit > best_fit:
                best, best_fit = dict(ind), fit

        order = np.argsort(fits)[::-1]
        logger.info(f"[Gen {gen}] tot nhat={fits[order[0]]:.4f} "
                    f"trung binh={np.mean(fits):.4f} | ky luc={best_fit:.4f}")

        if gen == args.generations - 1:
            break
        new_pop = [dict(pop[i]) for i in order[:args.elite]]        # elitism
        while len(new_pop) < args.pop:
            c1, c2 = crossover(tournament(pop, fits, rng),
                               tournament(pop, fits, rng), rng)
            new_pop.append(mutate(c1, rng, args.mutation_rate))
            if len(new_pop) < args.pop:
                new_pop.append(mutate(c2, rng, args.mutation_rate))
        pop = new_pop

    hp = decode(best)
    out = {"macro_f1": best_fit, "hyperparameters": hp,
           "proxy": {"clients": args.clients, "epochs": args.proxy_epochs,
                     "n_train": len(ytr), "n_val": len(yva)}}
    path = os.path.join(args.out_dir, "ga_best.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Tot nhat: macro_f1={best_fit:.4f} | {hp}\nLuu -> {path}")
    logger.info(
        "Dung cho main.py:  --lr %s --batch_size %s --local_ep %s --dropout %s "
        "--width %d %d --grid_size %s"
        % (hp["lr"], hp["batch_size"], hp.get("epochs", 1), hp["dropout"],
           hp["width"][0], hp["width"][1], hp["grid_size"]))
    logger.info("Gen chi dung trong GA (chua noi vao main.py): optimizer=%s "
                "momentum=%s l2=%s personal_coef=%s"
                % (hp.get("optimizer"), hp.get("momentum"), hp.get("l2"),
                   hp.get("personal_coef")))


if __name__ == "__main__":
    main()
