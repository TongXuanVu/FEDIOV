"""P2 - FedIoV: Flower server voi tong hop Byzantine-robust (Multi-Krum, m=5).

Bai bao: Heidari, Rastegar, Khonsari, FGCS 181 (2026).

Multi-Krum (Blanchard et al., NIPS 2017):
  1. Voi moi update w_i, tinh diem  s_i = tong binh phuong khoang cach toi
     (n - f - 2) update gan nhat.
  2. Chon m update co s_i nho nhat, lay trung binh co trong so theo so mau.

LUU Y KHI TAI HIEN: bai bao noi dung DONG THOI secure aggregation va Multi-Krum.
Hai co che nay xung dot -- Krum can thay ban ro tung update de do khoang cach,
con secure aggregation che chinh cac update do. Cai o day theo huong thuc dung:
Multi-Krum tren ban ro, bao mat duong truyen dua vao TLS cua Flower
(--certificates). Day la mot diem bai bao khong lam ro.

Chay:
  python server_iov.py --rounds 30 --num-clients 10 --krum-m 5 --byzantine 2
  python server_iov.py --strategy fedavg --rounds 30    # tat Krum de doi chung
  python server_iov.py --mode test --ckpt out/checkpoints/latest.pth
"""
import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (FitRes, Parameters, Scalar, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402
from model_kanconv import KANConvNet, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ----------------------------------------------------------------------------
# Multi-Krum
# ----------------------------------------------------------------------------
def flatten(ndarrays) -> np.ndarray:
    return np.concatenate([a.ravel() for a in ndarrays]).astype(np.float64)


def multi_krum_select(updates: List[np.ndarray], n_byz: int, m: int) -> Tuple[List[int], np.ndarray]:
    """Tra ve (chi so cac client duoc chon, diem Krum cua tung client)."""
    n = len(updates)
    n_close = max(1, n - n_byz - 2)
    flat = np.stack(updates)
    # ma tran khoang cach binh phuong
    sq = np.sum(flat ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    scores = np.sort(d2, axis=1)[:, :n_close].sum(axis=1)
    m = max(1, min(m, n))
    chosen = np.argsort(scores)[:m].tolist()
    return chosen, scores


def topsis_rank(updates: List[np.ndarray], mode: str = "consistency",
                param_weights=None):
    """TOPSIS tren cac ban cap nhat model — TANG MOT cua duong ong trong bai.

    Heidari et al., FGCS 181 (2026): "a two-stage aggregation pipeline that uses
    TOPSIS-based dimensionality reduction and Multi-Krum".

        Eq. (7)  m_ij = m_ij / sqrt( SUM_i m_ij^2 )      (chuan hoa theo cot)
        Eq. (8)  w_ij = w_j * m_ij
        C_i = S-_i / (S+_i + S-_i)

    HAI CACH DOC, va chung cho ket qua NGUOC NHAU:

    mode="literal"  — ap Eq. 7,8 thang len THAM SO THO nhu chu viet.
        Do thuc nghiem (20 client, 4 bi dau doc bien do gap 8 lan):
            C_i client that = 0.0525 | C_i client DOC = 0.7443
            client doc chiem max o 500/500 chieu -> chinh no LA nghiem ly tuong
            giu top-K theo C_i  =>  bat duoc 0/4 client doc
        Tuc la lam dung chu thi TOPSIS GIU LAI ke tan cong va vut client that.

    mode="consistency" (MAC DINH) — bam theo Y DINH ma bai mo ta bang loi:
        "ranking the incoming model updates based on their QUALITY AND
        CONSISTENCY" va "outlier ... lower similarity scores Ci ... reject".
        Tieu chi khong phai tung tham so ma la ba do do nhat quan:
            c1 = ||u_i - trung vi||          (chi phi: cang thap cang tot)
            c2 = | ||u_i|| - trung vi chuan | (chi phi)
            c3 = cosine(u_i, trung binh)      (loi ich: cang cao cang tot)
        Eq. 7,8 van duoc ap len ma tran tieu chi nay.

    Chon mode="literal" de tai hien y nguyen cong thuc; mode="consistency" de
    co hanh vi khop voi phan mo ta. Khac biet nay PHAI ghi trong bao cao.
    """
    M = np.stack(updates).astype(np.float64)                 # (N, D)
    if mode == "literal":
        crit = M
        loi_ich = np.ones(M.shape[1], dtype=bool)            # coi tat ca la benefit
    else:
        med = np.median(M, axis=0)
        mean = M.mean(axis=0)
        nrm = np.linalg.norm(M, axis=1)
        cos = (M @ mean) / (nrm * np.linalg.norm(mean) + 1e-12)
        crit = np.column_stack([
            np.linalg.norm(M - med, axis=1),                 # c1 chi phi
            np.abs(nrm - np.median(nrm)),                    # c2 chi phi
            cos,                                             # c3 loi ich
        ])
        loi_ich = np.array([False, False, True])

    col = np.linalg.norm(crit, axis=0)                       # Eq. (7)
    col[col == 0.0] = 1.0
    Mn = crit / col
    w = (np.full(crit.shape[1], 1.0 / crit.shape[1]) if param_weights is None
         else np.asarray(param_weights, dtype=np.float64))
    W = Mn * w                                               # Eq. (8)
    # tieu chi loi ich: ly tuong = max; tieu chi chi phi: ly tuong = min
    a_plus = np.where(loi_ich, W.max(axis=0), W.min(axis=0))
    a_minus = np.where(loi_ich, W.min(axis=0), W.max(axis=0))
    s_plus = np.linalg.norm(W - a_plus, axis=1)
    s_minus = np.linalg.norm(W - a_minus, axis=1)
    return s_minus / (s_plus + s_minus + 1e-12)


def cluster_by_rank(ci: np.ndarray, n_clusters: int) -> List[List[int]]:
    """Gom cum theo thu hang C_i — TANG GIUA cua duong ong trong bai.

    "The server groups the model updates into clusters of similar performance
    and behavior, depending on the [ranking]" (ngay truoc Eq. 14).

    Bai KHONG cho thuat toan gom cum, cung khong cho |C|. O day: sap theo C_i
    giam dan roi cat thanh n_clusters doan lien tiep — cach doc sat nhat voi
    "clusters of similar performance ... depending on the ranking". Day la
    LUA CHON CAI DAT, phai ghi trong bao cao.
    """
    order = np.argsort(-ci)
    return [g.tolist() for g in np.array_split(order, max(1, n_clusters))
            if len(g) > 0]


def auto_clusters(n: int) -> int:
    """|C| khong co trong bai. Phan phan tich do phuc tap chi noi "C grows with
    N", nen ta lay round(sqrt(N)) — cum to dan cham hon so client."""
    return max(1, int(round(np.sqrt(max(n, 1)))))


class MultiKrumStrategy(fl.server.strategy.FedAvg):
    """FedAvg + loc Byzantine bang Multi-Krum + checkpoint moi round."""

    def __init__(self, model, ckpt_dir: str, start_round: int = 0,
                 krum_m: int = 5, n_byzantine: int = 2, use_krum: bool = True,
                 use_topsis: bool = True, topsis_keep: float = 0.8,
                 topsis_mode: str = "consistency", n_clusters: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.start_round = start_round
        self.krum_m = krum_m
        self.n_byzantine = n_byzantine
        self.use_krum = use_krum
        self.use_topsis = use_topsis
        self.topsis_keep = topsis_keep
        self.topsis_mode = topsis_mode
        self.n_clusters = n_clusters          # 0 = tu tinh theo auto_clusters()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} client loi")

        metrics: Dict[str, Scalar] = {}
        kept = results
        ci = None
        n_cum = (self.n_clusters if self.n_clusters > 0
                 else auto_clusters(len(results)))

        # ---- TANG 1: loc bang TOPSIS (Eq. 7, 8) ----
        # Van tinh C_i khi tat TOPSIS neu con can thu hang de gom cum (Eq. 14).
        if len(results) > 2 and (self.use_topsis or n_cum > 1):
            flats0 = [flatten(parameters_to_ndarrays(r.parameters)) for _, r in results]
            ci = topsis_rank(flats0, self.topsis_mode)
        if self.use_topsis and ci is not None:
            k = max(2, int(round(self.topsis_keep * len(results))))
            keep_idx = sorted(np.argsort(-ci)[:k].tolist())
            flagged0 = [i for i, (_, r) in enumerate(results)
                        if r.metrics.get("attack", "none") != "none"]
            bo = set(range(len(results))) - set(keep_idx)
            metrics["topsis_kept"] = len(keep_idx)
            metrics["topsis_ci_min"] = float(ci.min())
            metrics["topsis_ci_max"] = float(ci.max())
            if flagged0:
                metrics["topsis_catch_rate"] = len(set(flagged0) & bo) / len(flagged0)
            bat = (f" | bat {len(set(flagged0) & bo)}/{len(flagged0)} client doc"
                   if flagged0 else "")
            logger.info(f"[Round {server_round}] TOPSIS giu {len(keep_idx)}/"
                        f"{len(results)} | C_i {ci.min():.4f}..{ci.max():.4f}{bat}")
            results = [results[i] for i in keep_idx]
            ci = ci[keep_idx]
            kept = results

        # ---- TANG 2: gom cum -> Multi-Krum trong tung cum -> Eq. 14 ----
        params = None
        if self.use_krum and len(results) > 2:
            ndarrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
            flats = [flatten(a) for a in ndarrays]
            # RANG BUOC BAI KHONG NOI: Multi-Krum giu m update MOI CUM, nen
            # cum phai DONG hon m, khong thi m >= |cum| va tang 2 khong loc gi
            # het. Do duoc: |C|=4 tren 16 update (cum 4 nguoi, m=5) -> giu
            # 16/16, ke tan cong con sot sau TOPSIS di thang vao model global
            # (bat 0/1, trong khi mot cum duy nhat bat 1/1).
            tran = max(1, len(results) // (self.krum_m + 1))
            if n_cum > tran:
                logger.info(f"[Round {server_round}] |C| {n_cum} -> {tran} "
                            f"(cum phai > m={self.krum_m} thi Multi-Krum moi loc)")
                n_cum = tran
            cums = (cluster_by_rank(ci, n_cum) if (ci is not None and n_cum > 1)
                    else [list(range(len(results)))])

            chon_tat, diem_min, diem_max = [], np.inf, -np.inf
            trung_binh_cum = []                  # M_k = (1/m) SUM_{i in S_k} M_i
            for g in cums:
                if len(g) > 2:
                    f_byz = max(0, int(round(self.n_byzantine * len(g) / len(results))))
                    m_g = max(1, min(self.krum_m, len(g)))
                    loc, sc = multi_krum_select([flats[i] for i in g], f_byz, m_g)
                    diem_min = min(diem_min, float(sc.min()))
                    diem_max = max(diem_max, float(sc.max()))
                else:
                    loc = list(range(len(g)))
                idx = [g[j] for j in loc]
                chon_tat += idx
                trung_binh_cum.append(
                    [np.mean([ndarrays[i][li] for i in idx], axis=0)
                     for li in range(len(ndarrays[0]))])

            # Eq. (14): M_G = (1/|C|) SUM_k (1/m) SUM_{i in S_k} M_i
            # Trung binh KHONG trong so — bai khong nhan theo so mau nhu FedAvg.
            agg = [np.mean([tb[li] for tb in trung_binh_cum], axis=0)
                   for li in range(len(ndarrays[0]))]
            params = ndarrays_to_parameters(agg)

            rejected = [i for i in range(len(results)) if i not in set(chon_tat)]
            kept = [results[i] for i in chon_tat]
            flagged = [i for i, (_, r) in enumerate(results)
                       if r.metrics.get("attack", "none") != "none"]
            caught = len(set(flagged) & set(rejected))
            metrics["krum_kept"] = len(chon_tat)
            metrics["krum_rejected"] = len(rejected)
            metrics["n_clusters"] = len(cums)
            if np.isfinite(diem_min):
                metrics["krum_score_min"] = diem_min
                metrics["krum_score_max"] = diem_max
            if flagged:
                metrics["byz_detection_rate"] = caught / len(flagged)
                logger.info(f"[Round {server_round}] Multi-Krum bat duoc "
                            f"{caught}/{len(flagged)} client doc hai")
            logger.info(f"[Round {server_round}] {len(cums)} cum (Eq.14) | giu "
                        f"{len(chon_tat)}/{len(results)} update"
                        + (f" | diem Krum {diem_min:.3e}..{diem_max:.3e}"
                           if np.isfinite(diem_min) else ""))

        if params is None:                      # fedavg, hoac qua it client
            params, agg_metrics = super().aggregate_fit(server_round, kept, [])
            metrics.update(agg_metrics)

        losses = [(r.num_examples, r.metrics.get("train_loss", 0.0)) for _, r in kept]
        n_tot = sum(n for n, _ in losses) or 1
        metrics["train_loss"] = sum(n * l for n, l in losses) / n_tot
        metrics["num_clients"] = len(kept)
        # Trong che do simulation, logger cua client nam trong Ray actor va
        # KHONG chay ra file log cua server — nen Eq.15 phai duoc bao cao qua
        # metrics thi moi kiem chung duoc.
        n_ca_nhan = sum(int(r.metrics.get("personalized", 0)) for _, r in kept)
        metrics["personalized_clients"] = n_ca_nhan
        if n_ca_nhan:
            logger.info(f"[Round {server_round}] Eq.15: {n_ca_nhan}/{len(kept)} "
                        f"client tron voi model cuc bo round truoc")

        if params is not None:
            abs_round = self.start_round + server_round
            sd = C.ndarrays_to_state_dict(self.model, parameters_to_ndarrays(params))
            C.save_checkpoint(self.ckpt_dir, abs_round, sd,
                              extra={"train_loss": metrics.get("train_loss"),
                                     "krum_kept": metrics.get("krum_kept")})
        return params, metrics


    def aggregate_evaluate(self, server_round, results, failures):
        """Gop do chinh xac CUC BO tung xe — thuoc do cua Bang 6 cho Eq.15.

        Bai bao cao ca do LECH giua cac xe ("Accuracy variance >4x across
        vehicles" khi bo ca nhan hoa), nen ta ghi ca mean/std/min/max.
        """
        if not results:
            return None, {}
        acc = np.array([float(r.metrics.get("accuracy", 0.0)) for _, r in results])
        n = np.array([max(r.num_examples, 0) for _, r in results], dtype=float)
        if n.sum() == 0:
            return None, {}
        loss = float((n * np.array([r.loss for _, r in results])).sum() / n.sum())
        m = {"local_acc_mean": float((acc * n).sum() / n.sum()),
             "local_acc_std": float(acc.std()),
             "local_acc_min": float(acc.min()),
             "local_acc_max": float(acc.max())}
        logger.info(f"[Round {server_round}] cuc bo tren {len(results)} xe: "
                    f"acc {m['local_acc_mean']:.4f} +/- {m['local_acc_std']:.4f} "
                    f"(min {m['local_acc_min']:.4f}, max {m['local_acc_max']:.4f})")
        # Ghi rieng ra CSV: cot nay khong nam trong METRIC_KEYS cua metrics.csv
        # nen khong lam hong collect_results.py.
        # Viet truc tiep, KHONG dung C.append_csv_row: ham do co dinh
        # CSV_HEADER cua metrics chung, va common.py la file dung chung ca 4
        # repo — them tham so vao do se lam 4 ban lech nhau.
        path = os.path.join(os.path.dirname(os.path.abspath(self.ckpt_dir)),
                            "local_metrics.csv")
        moi = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if moi:
                w.writerow(["round", "n_clients", "local_loss", "acc_mean",
                            "acc_std", "acc_min", "acc_max"])
            w.writerow([self.start_round + server_round, len(results),
                        round(loss, 6)]
                       + [round(m[k], 6) for k in
                          ("local_acc_mean", "local_acc_std",
                           "local_acc_min", "local_acc_max")])
        return loss, m


# ----------------------------------------------------------------------------
def make_evaluate_fn(model, loader, criterion, device, csv_file, out_dir,
                     class_names, total_rounds, start_round, task, cm_every=0):
    def evaluate_fn(server_round: int, parameters, config):
        if server_round == 0:
            return None
        abs_round = start_round + server_round
        model.load_state_dict(C.ndarrays_to_state_dict(model, parameters))
        model.to(device)
        m, y_true, y_pred = C.evaluate(model, loader, criterion, device)
        C.log_and_save_metrics(abs_round, m, csv_file)
        # Ghi confusion matrix o cuoi task, VA dinh ky neu bat --cm-every,
        # de bi cat giua chung van con ban gan nhat.
        if server_round == total_rounds or (cm_every and abs_round % cm_every == 0):
            tag = f"task{task}" if task is not None else "final"
            C.save_confusion_matrix(y_true, y_pred, out_dir, tag, class_names)
        return m["loss"], {k: v for k, v in m.items() if k != "loss"}
    return evaluate_fn


def fit_config_fn(local_epochs: int, lr: float):
    def fn(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round, "local_epochs": local_epochs, "lr": lr}
    return fn


def run_test(args, model, device):
    ckpt = args.ckpt or os.path.join(args.out_dir, "checkpoints", "latest.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt}")
    rnd, _ = C.load_checkpoint(ckpt, model)
    model.to(device)
    logger.info(f"Nap checkpoint {ckpt} (round {rnd})")
    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    m, y_true, y_pred = C.evaluate(model, loader, nn.CrossEntropyLoss(), device)
    logger.info(C.format_metrics(rnd, m))
    C.append_csv_row(os.path.join(args.out_dir, "test_metrics.csv"),
                     [rnd] + [round(m[k], 6) for k in C.METRIC_KEYS])
    tag = f"test_task{args.task}" if args.task is not None else "test"
    C.save_confusion_matrix(y_true, y_pred, args.out_dir, tag,
                            C.load_class_names(args.data_dir))


def main():
    p = argparse.ArgumentParser(description="P2 FedIoV Flower server (Multi-Krum)")
    p.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    p.add_argument("--strategy", choices=["multikrum", "fedavg"], default="multikrum")
    p.add_argument("--no-topsis", action="store_true",
                   help="Tat tang loc TOPSIS (chi con Multi-Krum) — de do dong "
                        "gop cua tung tang")
    p.add_argument("--topsis-mode", choices=["consistency", "literal"],
                   default="consistency",
                   help="literal = ap Eq.7,8 thang len tham so tho (dung chu bai, "
                        "nhung do duoc la GIU LAI ke tan cong). consistency = theo "
                        "y dinh mo ta bang loi")
    p.add_argument("--topsis-keep", type=float, default=0.8,
                   help="Ty le update giu lai sau TOPSIS")
    p.add_argument("--clusters", type=int, default=0,
                   help="|C| cua Eq.14. 0 = tu tinh round(sqrt(N)). 1 = mot cum "
                        "(Multi-Krum phang, KHONG theo Eq.14)")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--krum-m", type=int, default=5,
                   help="So update giu lai moi round (bai bao: m=5/cum)")
    p.add_argument("--byzantine", type=int, default=2,
                   help="So client doc hai GIA DINH (tham so f cua Krum)")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--width", type=int, nargs=2, default=[16, 32])
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--basis", choices=["fourier", "spline"],
                   default="fourier",
                   help="Co so ham cua lop KAN. fourier = dung bai "
                        "(Eq.16: 'a Fourier-based encoding'); "
                        "spline = ban cu (efficient-kan)")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--address", type=str, default="0.0.0.0:8082")
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--cm-every", type=int, default=0,
                   help="Ghi confusion matrix moi N round (0 = chi cuoi task)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(args.out_dir, "server.log"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiet bi: {device} | che do: {args.mode} | "
                f"strategy: {args.strategy} | task: {args.task}")

    model = KANConvNet(INPUT_LEN, NUM_GLOBAL_CLASSES, args.dropout,
                       tuple(args.width), args.grid_size, args.spline_order).to(device)
    logger.info(f"KANConvNet params: "
                f"{sum(q.numel() for q in model.parameters() if q.requires_grad):,}")

    if args.mode == "test":
        run_test(args, model, device)
        return

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    start_round = 0
    if args.mode == "resume":
        ckpt = args.ckpt or os.path.join(ckpt_dir, "latest.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Khong co checkpoint de resume: {ckpt}")
        start_round, _ = C.load_checkpoint(ckpt, model)
        model.to(device)
        logger.info(f"Resume tu round {start_round} ({ckpt})")

    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    class_names = C.load_class_names(args.data_dir)
    suffix = f"_task{args.task}" if args.task is not None else ""
    csv_file = os.path.join(args.out_dir, f"metrics{suffix}.csv")

    strategy = MultiKrumStrategy(
        model=model,
        ckpt_dir=ckpt_dir,
        start_round=start_round,
        krum_m=args.krum_m,
        n_byzantine=args.byzantine,
        use_krum=(args.strategy == "multikrum"),
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(args.num_clients * args.fraction_fit)),
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=ndarrays_to_parameters(C.get_model_parameters(model)),
        n_clusters=args.clusters,
        on_fit_config_fn=fit_config_fn(args.local_epochs, args.lr),
        evaluate_fn=make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                     csv_file, args.out_dir, class_names,
                                     args.rounds, start_round, args.task, args.cm_every),
    )

    logger.info(f"Server lang nghe {args.address} | {args.rounds} round | "
                f"Krum m={args.krum_m}, f={args.byzantine} | CSV -> {csv_file}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    logger.info(f"Xong. Ket qua trong {args.out_dir}")


if __name__ == "__main__":
    main()
