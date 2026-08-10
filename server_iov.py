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
DEFAULT_OUT_DIR = r"C:\FederatedLearning\Rebuild-IOV\P2-FEDIOV\out"


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


class MultiKrumStrategy(fl.server.strategy.FedAvg):
    """FedAvg + loc Byzantine bang Multi-Krum + checkpoint moi round."""

    def __init__(self, model, ckpt_dir: str, start_round: int = 0,
                 krum_m: int = 5, n_byzantine: int = 2, use_krum: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.start_round = start_round
        self.krum_m = krum_m
        self.n_byzantine = n_byzantine
        self.use_krum = use_krum

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

        if self.use_krum and len(results) > 2:
            ndarrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
            chosen, scores = multi_krum_select(
                [flatten(a) for a in ndarrays], self.n_byzantine, self.krum_m)
            rejected = [i for i in range(len(results)) if i not in chosen]
            kept = [results[i] for i in chosen]
            # ai tu khai bao la doc hai (chi de do luong, khong dung de loc)
            flagged = [i for i, (_, r) in enumerate(results)
                       if r.metrics.get("attack", "none") != "none"]
            caught = len(set(flagged) & set(rejected))
            metrics["krum_kept"] = len(chosen)
            metrics["krum_rejected"] = len(rejected)
            metrics["krum_score_min"] = float(scores.min())
            metrics["krum_score_max"] = float(scores.max())
            if flagged:
                metrics["byz_detection_rate"] = caught / len(flagged)
                logger.info(f"[Round {server_round}] Multi-Krum bat duoc "
                            f"{caught}/{len(flagged)} client doc hai")
            logger.info(f"[Round {server_round}] giu {len(chosen)}/{len(results)} "
                        f"update | diem Krum {scores.min():.3e}..{scores.max():.3e}")

        params, agg_metrics = super().aggregate_fit(server_round, kept, [])
        metrics.update(agg_metrics)

        losses = [(r.num_examples, r.metrics.get("train_loss", 0.0)) for _, r in kept]
        n_tot = sum(n for n, _ in losses) or 1
        metrics["train_loss"] = sum(n * l for n, l in losses) / n_tot
        metrics["num_clients"] = len(kept)

        if params is not None:
            abs_round = self.start_round + server_round
            sd = C.ndarrays_to_state_dict(self.model, parameters_to_ndarrays(params))
            C.save_checkpoint(self.ckpt_dir, abs_round, sd,
                              extra={"train_loss": metrics.get("train_loss"),
                                     "krum_kept": metrics.get("krum_kept")})
        return params, metrics


# ----------------------------------------------------------------------------
def make_evaluate_fn(model, loader, criterion, device, csv_file, out_dir,
                     class_names, total_rounds, start_round, task):
    def evaluate_fn(server_round: int, parameters, config):
        if server_round == 0:
            return None
        abs_round = start_round + server_round
        model.load_state_dict(C.ndarrays_to_state_dict(model, parameters))
        model.to(device)
        m, y_true, y_pred = C.evaluate(model, loader, criterion, device)
        C.log_and_save_metrics(abs_round, m, csv_file)
        if server_round == total_rounds:
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
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--address", type=str, default="0.0.0.0:8082")
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--ckpt", type=str, default=None)
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
        on_fit_config_fn=fit_config_fn(args.local_epochs, args.lr),
        evaluate_fn=make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                     csv_file, args.out_dir, class_names,
                                     args.rounds, start_round, args.task),
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
