"""P2 - FedIoV: KANConvNet (Kolmogorov-Arnold convolution) cho CICIoV.

Bai bao: Heidari, Rastegar, Khonsari, "FedIoV: ...", Future Generation Computer
Systems 181 (2026).

Lop KANLinear cai theo efficient-kan (Blealtan): thay vi mot trong so vo huong,
moi canh (i -> j) la mot HAM hoc duoc = SiLU co trong so + to hop B-spline.

    y_j = sum_i [ w_base[j,i] * silu(x_i) + sum_g w_spline[j,i,g] * B_g(x_i) ]

KANConv1d = unfold thanh cac patch roi ap dung KANLinear len chieu patch,
dung cach lam cua Convolutional-KANs (Tepsich).

Input CICIoV: (B, 31) -> (B, 1, 31). 13 lop.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31


class KANLinear(nn.Module):
    """Lop KAN day du: nhanh base (SiLU) + nhanh spline."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=scale_noise /
                        math.sqrt(in_features * (grid_size + spline_order)))

    def b_splines(self, x):
        """x: (B, in) -> (B, in, grid_size + spline_order) he so B-spline bac k."""
        grid = self.grid                                   # (in, G + 2k + 1)
        x = x.unsqueeze(-1)                                # (B, in, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases.contiguous()

    def forward(self, x):
        # x: (B, in_features)
        base = F.linear(F.silu(x), self.base_weight)
        spl = F.linear(self.b_splines(x).view(x.size(0), -1),
                       self.spline_weight.view(self.out_features, -1))
        return base + spl


class FourierKANLinear(nn.Module):
    """Lop KAN dung co so FOURIER thay vi B-spline.

    Bai FedIoV, Eq. (16):  z1 = W1 . F(z0) + b1
    "A linear transformation coupled with a FOURIER-BASED encoding then applies
    the Kolmogorov-Arnold mapping"

    Trong ca bai, chu "Fourier" xuat hien DUNG MOT LAN (dong 735) va chu
    "spline" KHONG xuat hien lan nao. Ban truoc cua ta dung B-spline (kieu
    efficient-kan) — sai co so.

        y_j = sum_i [ w_base[j,i]*silu(x_i)
                      + sum_{k=1..G} ( a[j,i,k]*cos(k*x_i) + b[j,i,k]*sin(k*x_i) ) ]

    G = grid_size = so hai bien (harmonic). Nhanh silu giu lai giong ban KAN goc
    de lop khong chet khi he so Fourier con nho.
    """

    def __init__(self, in_features, out_features, grid_size=5, scale_noise=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        # (out, in, G, 2): chieu cuoi la (cos, sin)
        self.fourier_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size, 2))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.fourier_weight,
                        std=scale_noise / math.sqrt(in_features * grid_size))
        self.register_buffer("k", torch.arange(1, grid_size + 1).float())

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        # (B, in, G)
        ang = x.unsqueeze(-1) * self.k
        feat = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)
        four = F.linear(feat.reshape(x.size(0), -1),
                        self.fourier_weight.reshape(self.out_features, -1))
        return base + four


class KANConv1d(nn.Module):
    """Convolution 1D voi nhan la mot KANLinear (thay vi tich vo huong)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 grid_size=5, spline_order=3, basis="fourier"):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.k, self.stride, self.padding = kernel_size, stride, padding
        if basis == "fourier":
            self.kan = FourierKANLinear(in_ch * kernel_size, out_ch, grid_size)
        else:
            self.kan = KANLinear(in_ch * kernel_size, out_ch, grid_size, spline_order)

    def forward(self, x):
        # x: (B, C, L) -> patch (B, C*k, L_out) -> KAN -> (B, out_ch, L_out)
        b = x.size(0)
        patches = F.unfold(x.unsqueeze(-1), (self.k, 1),
                           padding=(self.padding, 0), stride=(self.stride, 1))
        d, lo = patches.size(1), patches.size(2)
        out = self.kan(patches.transpose(1, 2).reshape(b * lo, d))
        return out.view(b, lo, self.out_ch).transpose(1, 2).contiguous()


class KANConvNet(nn.Module):
    """Backbone cua FedIoV. Giu so kenh nho vi KAN dat hon conv thuong ~10x.

    (B,31) -> (B,1,31) -> KANConv 16 -> pool -> KANConv 32 -> pool -> GAP -> KAN head
    """

    def __init__(self, input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES,
                 dropout=0.15, width=(16, 32), grid_size=5, spline_order=3,
                 basis="fourier"):
        super().__init__()
        c1, c2 = width
        self.input_len = input_len
        self.num_classes = num_classes
        self.feat_dim = c2

        # Tanh sau moi BatchNorm: ep dau vao cua lop KAN ke tiep ve dung
        # grid_range=(-1,1). Neu bo Tanh, phan lon gia tri roi ngoai luoi ->
        # co so B-spline bang 0 va KAN thoai hoa thanh mot lop SiLU thuong.
        self.block1 = nn.Sequential(
            KANConv1d(1, c1, 3, padding=1, grid_size=grid_size,
                      spline_order=spline_order, basis=basis),
            nn.BatchNorm1d(c1), nn.Tanh(), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(
            KANConv1d(c1, c2, 3, padding=1, grid_size=grid_size,
                      spline_order=spline_order, basis=basis),
            nn.BatchNorm1d(c2), nn.Tanh(), nn.MaxPool1d(2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = (FourierKANLinear(c2, num_classes, grid_size) if basis == "fourier"
                     else KANLinear(c2, num_classes, grid_size, spline_order))

    def embed(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.tanh(x)                 # ep ve [-1,1] cho khop grid_range cua KAN
        x = self.block2(self.block1(x))
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


if __name__ == "__main__":
    m = KANConvNet()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
    print("Output shape:", m(torch.randn(4, INPUT_LEN)).shape)
