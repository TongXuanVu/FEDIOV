"""FedIoV — mot lenh chay het, dung CHE DO CUA BAI BAO.

Heidari, Rastegar, Khonsari, "FedIoV: A secure and adaptive federated framework
for real-time intrusion detection in vehicular networks", FGCS 181 (2026) 108448.

MAC DINH = dung bai nhat:
  --basis fourier    Eq. (16):  z1 = W1 . F(z0) + b1
                     "A linear transformation coupled with a FOURIER-BASED
                     encoding then applies the Kolmogorov-Arnold mapping"
  --strategy multikrum   Duong ong tong hop hai tang cua bai: loc bang TOPSIS
                     roi gom cum ben vung bang Multi-Krum
  full data          moi client dung het shard, danh gia tren het tap test

CHO LECH SO VOI BAI, phai ghi trong bao cao:
  1. Bai KHONG dinh nghia sigma_Kolmogorov bang cong thuc — chi goi ten. Ta
     dung nhanh SiLU + to hop Fourier (xem FourierKANLinear trong
     model_kanconv.py). Day la LUA CHON CAI DAT.
  2. Bai khong cho so kenh / so lop cu the. width=(16,32) la lua chon cua ta.
  3. Bang 3 cua bai cho KHONG GIAN TIM KIEM cua GA (lr, batch, epoch, momentum,
     dropout, optimizer, L2, he so ca nhan hoa, Pc, Pm) chu khong cho gia tri
     cuoi cung. Chay ga_search.py neu muon do lai; mac dinh o day khong chay GA.
     Tat ca 10 gen deu da noi vao duong chay chinh (--optimizer, --momentum,
     --l2, --personal_coef, ...) nen ket qua GA dung duoc ngay.
  5. |C| trong Eq. (14) khong co trong bai (chi noi "C grows with N"). Mac dinh
     |C| = round(sqrt(N)). Thuat toan gom cum cung khong co — ta cat theo thu
     hang C_i. Xem cluster_by_rank() trong server_iov.py.
  6. Eq. (15) can model cuc bo cua round TRUOC. Trong simulation, client bi tao
     lai moi round nen trang thai duoc luu ra <out_dir>/client_state/.
  4. Bai mo phong bang OMNeT++ / Veins / SUMO de sinh do tre, mat goi. Ta khong
     co, nen phan "adaptive" theo trang thai mang khong tai hien duoc.

Chay:
  python main.py --data_dir /kaggle/input/... --num_users 100
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser(description="FedIoV: mot lenh chay het")
    p.add_argument("--data_dir", required=True,
                   help="Thu muc chua federated_data/ va global_test_data.pt")
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    p.add_argument("--num_users", type=int, default=100)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--com_round", type=int, default=30, help="Round MOI task")
    p.add_argument("--local_ep", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--max_samples", type=int, default=0, help="0 = full data")
    p.add_argument("--test_samples", type=int, default=0,
                   help="0 = danh gia tren HET tap test moi round")
    # --- kien truc ---
    p.add_argument("--basis", choices=["fourier", "spline"], default="fourier",
                   help="fourier = dung bai (Eq.16). spline = ban cu")
    p.add_argument("--width", type=int, nargs=2, default=[16, 32])
    p.add_argument("--grid_size", type=int, default=5,
                   help="So hai bien Fourier (hoac so o luoi neu dung spline)")
    # --- tong hop ben vung ---
    p.add_argument("--strategy", choices=["multikrum", "fedavg"],
                   default="multikrum", help="multikrum = dung bai")
    p.add_argument("--no_topsis", action="store_true",
                   help="Tat tang loc TOPSIS, chi con Multi-Krum")
    p.add_argument("--topsis_mode", choices=["consistency", "literal"],
                   default="consistency",
                   help="literal = dung chu Eq.7,8 (do duoc: GIU LAI ke tan cong). "
                        "consistency = theo y dinh mo ta bang loi (mac dinh)")
    p.add_argument("--topsis_keep", type=float, default=0.8)
    p.add_argument("--clusters", type=int, default=0,
                   help="|C| cua Eq.(14). 0 = tu tinh round(sqrt(N)); 1 = mot "
                        "cum duy nhat (Multi-Krum phang, KHONG theo Eq.14)")
    # --- toi uu cuc bo (Bang 3) + ca nhan hoa (Eq.15) ---
    p.add_argument("--optimizer", default="AdamW",
                   choices=["AdamW", "RMSprop", "AdaDelta", "SGD", "Nadam", "Adam"],
                   help="omega trong Bang 3 cua bai")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="mu trong Bang 3 (chi tac dung voi SGD/RMSprop)")
    p.add_argument("--l2", type=float, default=0.0,
                   help="lambda trong Bang 3 = weight_decay. Bai co hang ablation "
                        "'w/o L2 Regularization'")
    p.add_argument("--personal_coef", type=float, default=1.0,
                   help="alpha cua Eq.(15): M_i = a*M_G + (1-a)*M_i. 1.0 = TAT "
                        "ca nhan hoa (= hang 'w/o Personalization' Bang 6). Bai "
                        "khong cong bo gia tri da dung; Bang 3 cho {0.15..0.95}")
    p.add_argument("--local_val", type=float, default=0.0,
                   help="Ty le shard client giu lam validation CUC BO. Dat >0 "
                        "(vd 0.1) de do do chinh xac TUNG XE — day moi la thuoc "
                        "do ma Bang 6 dung cho Eq.15. 0 = tat, chay nhanh hon")
    p.add_argument("--krum_m", type=int, default=5)
    p.add_argument("--byzantine", type=int, default=2,
                   help="So client doc ma Multi-Krum gia dinh")
    p.add_argument("--attack", choices=["none", "signflip", "gauss", "label"],
                   default="none")
    p.add_argument("--attack_ids", type=int, nargs="*", default=[])
    p.add_argument("--attack_scale", type=float, default=5.0)
    # --- van hanh ---
    p.add_argument("--cm_every", type=int, default=5)
    p.add_argument("--flat", action="store_true")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--fed_subdir", default="federated_data",
                   choices=["federated_data", "federated_data_fewshot",
                            "federated_data_10shot"])
    p.add_argument("--actor_gpus", type=float, default=-1.0)
    p.add_argument("--actor_cpus", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    argv = [
        "run_sim.py",
        "--data-dir", a.data_dir,
        "--out-dir", a.out_dir,
        "--clients", str(a.num_users),
        "--rounds", str(a.com_round),
        "--tasks", "none" if a.flat else ",".join(str(t) for t in range(a.tasks)),
        "--local-epochs", str(a.local_ep),
        "--batch-size", str(a.batch_size),
        "--lr", str(a.lr),
        "--dropout", str(a.dropout),
        "--max-samples", str(a.max_samples),
        "--test-samples", str(a.test_samples),
        "--cm-every", str(a.cm_every),
        "--seed", str(a.seed),
        "--basis", a.basis,
        "--width", str(a.width[0]), str(a.width[1]),
        "--grid-size", str(a.grid_size),
        "--strategy", a.strategy,
        "--topsis-mode", a.topsis_mode,
        "--topsis-keep", str(a.topsis_keep),
        "--clusters", str(a.clusters),
        "--optimizer", a.optimizer,
        "--momentum", str(a.momentum),
        "--l2", str(a.l2),
        "--personal-coef", str(a.personal_coef),
        "--local-val", str(a.local_val),
        "--fed-subdir", a.fed_subdir,
        "--krum-m", str(a.krum_m),
        "--byzantine", str(a.byzantine),
        "--attack", a.attack,
        "--attack-scale", str(a.attack_scale),
        "--actor-gpus", str(a.actor_gpus),
        "--actor-cpus", str(a.actor_cpus),
    ]
    if a.no_topsis:
        argv.append("--no-topsis")
    if a.attack_ids:
        argv += ["--attack-ids"] + [str(i) for i in a.attack_ids]
    if a.restart:
        argv.append("--restart")

    print("=" * 70)
    print("FedIoV | Heidari, Rastegar, Khonsari, FGCS 181 (2026) 108448")
    print(f"  du lieu   : {a.data_dir}")
    print(f"  ket qua   : {a.out_dir}")
    print(f"  cau hinh  : {a.num_users} client | {a.tasks} task x {a.com_round} "
          f"round = {a.tasks * a.com_round} round")
    co_so = ("Fourier (Eq.16 cua bai)" if a.basis == "fourier"
             else "B-spline (ban cu, KHONG theo bai)")
    print(f"  KANConvNet: co so {co_so}, width={tuple(a.width)}, "
          f"grid={a.grid_size}")
    tang1 = "tat" if a.no_topsis else f"TOPSIS ({a.topsis_mode}, giu {a.topsis_keep:.0%})"
    print(f"  tong hop  : hai tang | tang 1 = {tang1}")
    cum = ("tu tinh sqrt(N)" if a.clusters == 0 else
           ("1 cum — KHONG theo Eq.14" if a.clusters == 1 else f"{a.clusters} cum"))
    print(f"              tang 2 = {a.strategy}"
          f"{f' (m={a.krum_m}, byzantine={a.byzantine})' if a.strategy == 'multikrum' else ''}"
          f" | Eq.14: {cum}")
    ca_nhan = ("TAT (= hang 'w/o Personalization' Bang 6)" if a.personal_coef >= 1.0
               else f"alpha={a.personal_coef}")
    print(f"  Eq.15     : {ca_nhan}")
    print(f"  toi uu    : {a.optimizer} | momentum {a.momentum} | L2 {a.l2}")
    if a.attack != "none":
        print(f"  TAN CONG  : {a.attack} tren client {a.attack_ids or 'chua chi dinh'}")
    print("  LUU Y: sigma_Kolmogorov khong duoc bai dinh nghia bang cong thuc —")
    print("         cach cai dat o day la lua chon cua ta, xem model_kanconv.py")
    print("=" * 70, flush=True)

    sys.argv = argv
    import run_sim
    run_sim.main()


if __name__ == "__main__":
    main()
