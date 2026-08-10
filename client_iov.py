"""P2 - FedIoV: Flower client (KANConvNet, CICIoV).

Bai bao: Heidari, Rastegar, Khonsari, FGCS 181 (2026).

Client = xe/RSU trong cum. Ho tro mo phong client DOC HAI (Byzantine) de kiem
chung phan Multi-Krum o server:

  --attack none      : client trung thuc
  --attack signflip  : gui -scale * w  (dao dau gradient)
  --attack gauss     : gui nhieu Gauss thuan tuy
  --attack label     : hoan doi nhan (label flipping) roi train binh thuong

Chay:
  python client_iov.py --client-id 0
  python client_iov.py --client-id 7 --attack signflip
"""
import argparse
import logging
import os
import sys

import flwr as fl
import numpy as np
import torch
import torch.optim as optim

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)                      # dung chung common.py -> so sanh cong bang

import common as C                               # noqa: E402
from model_cnn1d import FocalLoss                # noqa: E402
from model_kanconv import KANConvNet, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"


class FedIoVClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir, device, max_samples, batch_size,
                 task, lr, dropout, width, grid_size, spline_order,
                 attack="none", attack_scale=5.0, seed=42):
        self.cid = client_id
        self.device = device
        self.lr = lr
        self.attack = attack
        self.attack_scale = attack_scale
        self.rng = np.random.default_rng(seed + client_id)

        x, y = C.load_client_data(data_dir, client_id, task, max_samples)
        if attack == "label":
            y = self._flip_labels(y)
        self.loader = C.make_loader(x, y, batch_size, shuffle=True)
        self.n_samples = len(y)

        self.model = KANConvNet(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout,
                                width, grid_size, spline_order).to(device)
        self.criterion = FocalLoss(alpha=C.make_focal_alpha(y).to(device), gamma=2.0)
        if attack != "none":
            logger.warning(f"[Client {self.cid}] DANG O CHE DO TAN CONG: {attack}")

    def _flip_labels(self, y):
        n_cls = C.NUM_GLOBAL_CLASSES
        return (n_cls - 1 - y).astype(np.int64)

    # ---- Flower API -------------------------------------------------------
    def get_parameters(self, config):
        return C.get_model_parameters(self.model)

    def set_parameters(self, parameters):
        self.model.load_state_dict(C.ndarrays_to_state_dict(self.model, parameters))

    def _poison(self, params):
        if self.attack == "signflip":
            return [-self.attack_scale * p for p in params]
        if self.attack == "gauss":
            return [self.rng.normal(0, 1, p.shape).astype(p.dtype) for p in params]
        return params

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        rnd = int(config.get("server_round", 0))
        lr = float(config.get("lr", self.lr))

        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=lr)
        total_loss, n_batches = 0.0, 0
        for _ in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                opt.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
        avg = total_loss / max(n_batches, 1)
        logger.info(f"[Client {self.cid}][Round {rnd}] {epochs} epoch, "
                    f"n={self.n_samples}, train_loss={avg:.4f}")
        params = self._poison(C.get_model_parameters(self.model))
        return params, self.n_samples, {"train_loss": avg, "attack": self.attack}

    def evaluate(self, parameters, config):
        return 0.0, self.n_samples, {}


def main():
    p = argparse.ArgumentParser(description="P2 FedIoV Flower client")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--server", type=str, default="127.0.0.1:8082")
    p.add_argument("--max-samples", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--width", type=int, nargs=2, default=[16, 32])
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--attack", choices=["none", "signflip", "gauss", "label"],
                   default="none")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    C.setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = FedIoVClient(args.client_id, args.data_dir, device, args.max_samples,
                          args.batch_size, args.task, args.lr, args.dropout,
                          tuple(args.width), args.grid_size, args.spline_order,
                          args.attack, args.attack_scale, args.seed)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
