"""P2 - FedIoV: Flower client (KANConvNet, CICIoV).

Bai bao: Heidari, Rastegar, Khonsari, FGCS 181 (2026).

Client = xe/RSU trong cum. Ho tro mo phong client DOC HAI (Byzantine) de kiem
chung phan Multi-Krum o server:

  --attack none      : client trung thuc
  --attack signflip  : gui -scale * w  (dao dau gradient)
  --attack gauss     : gui nhieu Gauss thuan tuy
  --attack label     : hoan doi nhan (label flipping) roi train binh thuong

CA NHAN HOA (Eq. 15 cua bai):
    M_i(t+1) = alpha * M_G(t) + (1 - alpha) * M_i(t)
Moi xe tron model global vua nhan voi model CUA CHINH NO o round truoc, roi
moi train. alpha=1 => bo qua ca nhan hoa (dung hang "w/o Personalization"
trong Bang 6). Bai KHONG cong bo gia tri alpha da dung, Bang 3 chi cho khong
gian tim {0.15, 0.35, 0.55, 0.75, 0.95}.

Trang thai cuc bo phai LUU RA DIA (--state-dir): trong che do simulation cua
Flower, doi tuong client bi tao lai moi round nen bien thanh vien khong song
sot. Da do bang thuc nghiem o P4 (--jitter tro thanh vo tac dung am tham).

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

# Bang 3 cua bai cho omega in {AdamW, RMSprop, AdaDelta, SGD, Nadam}.
# "Adam" khong co trong bai; giu lai de doi chung voi ban truoc.
OPTIMIZERS = ["AdamW", "RMSprop", "AdaDelta", "SGD", "Nadam", "Adam"]


def make_optimizer(params, name="AdamW", lr=1e-3, momentum=0.9, l2=0.0):
    """Dung optimizer theo ten trong Bang 3. l2 = weight_decay (bai co hang
    ablation 'w/o L2 Regularization' nen day la thanh phan co ten trong bai)."""
    n = (name or "AdamW").lower()
    if n == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=l2)
    if n == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=l2)
    if n == "adadelta":
        return optim.Adadelta(params, lr=lr, weight_decay=l2)
    if n == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=l2)
    if n == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=l2)
    if n == "adam":
        return optim.Adam(params, lr=lr, weight_decay=l2)
    raise ValueError(f"Optimizer khong biet: {name}. Chon trong {OPTIMIZERS}")


class FedIoVClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir, device, max_samples, batch_size,
                 task, lr, dropout, width, grid_size, spline_order,
                 basis="fourier", attack="none", attack_scale=5.0, seed=42,
                 optimizer="AdamW", momentum=0.9, l2=0.0,
                 personal_coef=1.0, state_dir=None, local_val=0.0):
        self.cid = client_id
        self.device = device
        self.lr = lr
        self.attack = attack
        self.attack_scale = attack_scale
        self.rng = np.random.default_rng(seed + client_id)
        self.optimizer = optimizer
        self.momentum = momentum
        self.l2 = l2
        self.personal_coef = float(personal_coef)
        self.state_dir = state_dir
        self.task = task
        self.da_tron = False

        x, y = C.load_client_data(data_dir, client_id, task, max_samples)
        if attack == "label":
            y = self._flip_labels(y)
        # Tach mot phan lam validation CUC BO. Bang 6 cua bai danh gia ca nhan
        # hoa bang do chinh xac TUNG XE ("Accuracy variance >4x across
        # vehicles"), khong phai bang model global — nen phai co tap nay thi
        # Eq.15 moi do duoc.
        self.val_loader, self.n_val = None, 0
        if local_val > 0 and len(y) >= 20:
            k = max(1, int(len(y) * local_val))
            r = np.random.default_rng(seed + client_id)
            idx = r.permutation(len(y))
            vi, ti = idx[:k], idx[k:]
            self.val_loader = C.make_loader(x[vi], y[vi], 4096, shuffle=False)
            self.n_val = len(vi)
            x, y = x[ti], y[ti]
        self.loader = C.make_loader(x, y, batch_size, shuffle=True)
        self.n_samples = len(y)

        self.model = KANConvNet(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout,
                                width, grid_size, spline_order, basis).to(device)
        self.criterion = FocalLoss(alpha=C.make_focal_alpha(y).to(device), gamma=2.0)
        if attack != "none":
            logger.warning(f"[Client {self.cid}] DANG O CHE DO TAN CONG: {attack}")

    def _flip_labels(self, y):
        n_cls = C.NUM_GLOBAL_CLASSES
        return (n_cls - 1 - y).astype(np.int64)

    # ---- Flower API -------------------------------------------------------
    def get_parameters(self, config):
        return C.get_model_parameters(self.model)

    # ---- trang thai cuc bo cho Eq. 15 ---------------------------------------
    def _state_path(self):
        ten_task = "flat" if self.task is None else f"task{self.task}"
        return os.path.join(self.state_dir, f"client_{self.cid}_{ten_task}.npz")

    def _load_local(self):
        p = self._state_path()
        if not os.path.exists(p):
            return None
        try:
            with np.load(p) as z:
                return [z[f"a{i}"] for i in range(len(z.files))]
        except Exception as e:                     # file hong -> coi nhu chua co
            logger.warning(f"[Client {self.cid}] khong doc duoc {p}: {e}")
            return None

    def _save_local(self, params):
        os.makedirs(self.state_dir, exist_ok=True)
        np.savez(self._state_path(), **{f"a{i}": a for i, a in enumerate(params)})

    def set_parameters(self, parameters):
        """Eq. (15): M_i(t+1) = alpha*M_G(t) + (1-alpha)*M_i(t).

        alpha = 1 -> chi dung global (bo ca nhan hoa). Round dau chua co model
        cuc bo nen luon dung global nguyen ven.
        """
        self.da_tron = False
        if self.personal_coef < 1.0 and self.state_dir:
            truoc = self._load_local()
            if truoc is not None and len(truoc) == len(parameters):
                a = self.personal_coef
                parameters = [a * g + (1.0 - a) * l
                              for g, l in zip(parameters, truoc)]
                self.da_tron = True
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
        opt = make_optimizer(self.model.parameters(), self.optimizer, lr,
                             self.momentum, self.l2)
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
        sach = C.get_model_parameters(self.model)
        if self.personal_coef < 1.0 and self.state_dir:
            self._save_local(sach)          # luu ban THAT, khong luu ban da dau doc
        tron = " | Eq.15 da tron" if self.da_tron else ""
        logger.info(f"[Client {self.cid}][Round {rnd}] {epochs} epoch, "
                    f"n={self.n_samples}, train_loss={avg:.4f}{tron}")
        return (self._poison(sach), self.n_samples,
                {"train_loss": avg, "attack": self.attack,
                 "personalized": int(self.da_tron)})

    def evaluate(self, parameters, config):
        """Danh gia model DA CA NHAN HOA tren tap val cuc bo cua chinh xe nay.

        set_parameters() ap Eq.15 truoc, nen thu duoc do la M_i(t+1) — dung
        model ma xe se dung that, khong phai model global.
        """
        if self.val_loader is None:
            return 0.0, 0, {}
        import torch as _t
        self.set_parameters(parameters)
        self.model.eval()
        dung, tong, mat, nb = 0, 0, 0.0, 0
        with _t.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                out = self.model(xb)
                mat += float(self.criterion(out, yb)); nb += 1
                dung += int((out.argmax(1) == yb).sum()); tong += len(yb)
        acc = dung / max(tong, 1)
        return mat / max(nb, 1), tong, {"accuracy": acc,
                                        "personalized": int(self.da_tron)}


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
    p.add_argument("--basis", choices=["fourier", "spline"],
                   default="fourier",
                   help="Co so ham cua lop KAN. fourier = dung bai "
                        "(Eq.16: 'a Fourier-based encoding'); "
                        "spline = ban cu (efficient-kan)")
    p.add_argument("--attack", choices=["none", "signflip", "gauss", "label"],
                   default="none")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="AdamW",
                   help="Bang 3 cua bai (omega)")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="Chi co tac dung voi SGD/RMSprop")
    p.add_argument("--l2", type=float, default=0.0, help="weight_decay")
    p.add_argument("--personal-coef", type=float, default=1.0,
                   help="alpha cua Eq.15. 1.0 = tat ca nhan hoa")
    p.add_argument("--state-dir", type=str, default=None,
                   help="Noi luu model cuc bo cho Eq.15")
    p.add_argument("--local-val", type=float, default=0.0,
                   help="Ty le du lieu client giu lai lam validation CUC BO. "
                        ">0 moi do duoc tac dung cua Eq.15 (Bang 6 cua bai)")
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    C.setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = FedIoVClient(args.client_id, args.data_dir, device, args.max_samples,
                          args.batch_size, args.task, args.lr, args.dropout,
                          tuple(args.width), args.grid_size, args.spline_order,
                          args.basis, args.attack, args.attack_scale, args.seed,
                          args.optimizer, args.momentum, args.l2,
                          args.personal_coef, args.state_dir, args.local_val)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
