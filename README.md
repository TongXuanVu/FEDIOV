# FedIoV — tái hiện trên CICIoV

> Heidari, Rastegar, Khonsari, *FedIoV*, Future Generation Computer Systems 181 (2026)

Tái hiện trên **CICIoV** (31 đặc trưng, 13 lớp) để so sánh được với ba bài còn
lại và với AFSIC-IoV / FedLiTeCAN trên cùng một backbone.

Ba repo anh em: [VANFED-IDS](https://github.com/TongXuanVu/VANFED-IDS) ·
[FEDIOV](https://github.com/TongXuanVu/FEDIOV) · [IOVFD](https://github.com/TongXuanVu/IOVFD) ·
[SDNFL-IDS](https://github.com/TongXuanVu/SDNFL-IDS)

---

## Ý tưởng được tái hiện

| Thành phần của bài | Ở đâu | Trạng thái |
|---|---|---|
| KANConvNet, cơ sở **Fourier** (Eq. 16) | `model_kanconv.py` | ✅ mặc định |
| Tầng 1 — **TOPSIS** xếp hạng update (Eq. 7-13) | `server_iov.py: topsis_rank` | ✅ |
| Gom cụm theo thứ hạng rồi **Multi-Krum** trong từng cụm | `server_iov.py: cluster_by_rank` | ✅ |
| Tổng hợp **Eq. 14** — trung bình không trọng số các cụm | `MultiKrumStrategy.aggregate_fit` | ✅ |
| **Cá nhân hoá Eq. 15** `M_i = αM_G + (1−α)M_i` | `client_iov.py: set_parameters` | ✅ (mặc định tắt, xem dưới) |
| **GA** dò siêu tham số, không gian đúng Bảng 3 | `ga_search.py` | ✅ 10 gen |
| Optimizer / momentum / L2 của Bảng 3 | `client_iov.py: make_optimizer` | ✅ |
| Kênh mã hoá, OMNeT++/Veins/SUMO | — | ❌ không có

## Cài đặt

```bash
git clone https://github.com/TongXuanVu/FEDIOV.git
cd FEDIOV
pip install -r requirements.txt
```

### Trên Kaggle — clone đúng repo này là chạy được

```python
!git clone -q https://github.com/TongXuanVu/FEDIOV.git /kaggle/working/FEDIOV
!pip install -q flwr
CODE = "/kaggle/working/FEDIOV"
DATA = "/kaggle/input/iov-100client"      # Kaggle Dataset chứa federated_data/

!cd {CODE} && python run_fl.py --data-dir {DATA} --clients 10 --rounds 20
```

Kaggle đã có sẵn torch / numpy / scikit-learn / matplotlib, chỉ thiếu `flwr`.
Repo này không phụ thuộc ba repo kia — không cần clone thêm gì.

## Chạy — một lệnh

```bash
python main.py --data_dir <DATA> --num_users 100
```

Mặc định đã là **chế độ của bài**: cơ sở Fourier, TOPSIS → gom cụm → Multi-Krum
→ Eq. 14, full data, đánh giá trên hết tập test mỗi round.

Bật cá nhân hoá Eq. 15 và đo đúng thước đo của Bảng 6:

```bash
python main.py --data_dir <DATA> --personal_coef 0.55 --local_val 0.15
```

Dùng kết quả GA (`ga_best.json` in sẵn dòng lệnh ở trường `main_py_args`):

```bash
python main.py --data_dir <DATA> --optimizer Nadam --momentum 0.95 --l2 0.0005 \
               --lr 0.005 --batch_size 48 --local_ep 7 --personal_coef 0.55
```

## Chạy tay

Cần **1 server + N client**. Muốn chạy tay thì mở N+1 terminal, server trước:

```bash
python server_iov.py --rounds 30 --num-clients 10 --data-dir <DATA>
python client_iov.py --client-id 0 --data-dir <DATA>
python client_iov.py --client-id 1 --data-dir <DATA>
```

Trên Kaggle/Colab không mở được nhiều terminal — dùng `run_fl.py`, nó tự sinh
server + N client và chạy nối tiếp task 0→4 (class-incremental), resume giữa
các task nên số round liên tục:

```bash
python run_fl.py --data-dir <DATA> --clients 10 --rounds 20
python run_fl.py --data-dir <DATA> --tasks none      # FL thường, gộp cả 5 task
```

Tìm siêu tham số trước (GA), rồi truyền kết quả vào server/client:

```bash
python ga_search.py --pop 12 --generations 6 --clients 0 1 2 3 --data-dir <DATA>
```

Bật một client độc hại để kiểm chứng Multi-Krum:

```bash
python client_iov.py --client-id 7 --attack signflip --data-dir <DATA>
```

Tắt Krum để đối chứng:

```bash
python run_fl.py --data-dir <DATA> --server-extra="--strategy fedavg"
```


> `--server-extra` và `--client-extra` **bắt buộc viết dạng có dấu `=`**.
> Viết cách ra sẽ lỗi `expected one argument` vì argparse tưởng là option mới.

### Ba chế độ

```bash
python server_iov.py --mode train  --rounds 30
python server_iov.py --mode resume --rounds 50           # chạy tiếp từ latest.pth
python server_iov.py --mode test   --ckpt out/checkpoints/latest.pth
```

## Dữ liệu

Định dạng khớp AFSIC-IoV:

```
<DATA>/federated_data/client_<id>_task_<t>.pt    # t = 1..5, dict {'x','y'}
<DATA>/global_test_data.pt
<DATA>/class_mapping.json
```

Chia lớp theo task: `TASK_INCREMENTS = [3, 3, 3, 2, 2]` (13 lớp / 5 task).
`run_fl.py` tự bỏ qua client thiếu file của task đang chạy thay vì để server
treo chờ mãi.

## Kết quả

Đổ vào `--out-dir` (mặc định `out/`):

| File | Nội dung |
|---|---|
| `metrics_task*.csv` | 1 dòng/round, 12 cột: loss, accuracy, micro/macro/weighted P-R-F1 |
| `confusion_matrix_task*.csv` / `_normalized.csv` / `.png` | cuối mỗi task |
| `classification_report_task*.txt` | P/R/F1 từng lớp |
| `checkpoints/round_NNN.pth`, `latest.pth` | resume được |
| `local_metrics.csv` | chỉ khi `--local_val > 0`: acc từng xe (mean/std/min/max) — thước đo của Eq. 15 |
| `client_state/` | model cục bộ từng client cho Eq. 15 |
| `ga_history.csv`, `ga_best.json` | quá trình và kết quả GA (`main_py_args` là dòng lệnh dùng ngay được) |

Gộp nhiều lần chạy + đo mức độ quên:

```bash
python collect_results.py --runs A=out_a B=out_b --out-dir ket_qua
```

Sinh `comparison.csv`, ma trận quên từng lần chạy (`forgetting_*.csv`), và
`accuracy_curve.png`. Mức độ quên tính theo định nghĩa chuẩn class-incremental:
`forgetting(j) = max_{t<T} acc(j,t) − acc(j,T)`.

## Kiểm thử

```bash
python smoke_test.py
```

Tự sinh dữ liệu giả đúng định dạng, chạy 2 round, kiểm CSV đủ 12 cột, checkpoint
nạp lại được, confusion matrix có sinh ra, và cả `--mode test` lẫn `--mode resume`.

**Trạng thái:** đã chạy thật trên **dữ liệu giả**, chưa lần nào trên CICIoV thật.

Đo trên 20 client / 5 client độc / 3 round, đường ống đầy đủ (TOPSIS giữ 80% →
gom cụm → Multi-Krum m=5):

| kiểu tấn công | TOPSIS bắt | Multi-Krum bắt nốt | tổng |
|---|---|---|---|
| signflip | 4/5 | 1/1 | **5/5** |
| gauss | 4/5 | 1/1 | **5/5** |
| label | 1/5 | 4/4 | **5/5** |

Hai tầng bù nhau: TOPSIS mạnh với tấn công biên độ, Multi-Krum mới là thứ bắt
được đầu độc nhãn. Bỏ một trong hai đều để lọt.

## Khác gì so với bài báo

Bài gốc **không công bố source code**. Mọi con số phải tự đo lại, không kỳ vọng
khớp bảng kết quả trong bài.

### 1. Secure aggregation và Multi-Krum xung đột nhau

Krum cần thấy **bản rõ** từng update để đo khoảng cách, secure aggregation lại
che chính các update đó. Bài nói dùng cả hai. Ở đây Multi-Krum chạy trên bản rõ,
bảo mật đường truyền giao cho TLS của Flower.

### 2. TOPSIS: chữ và ý trong bài ngược nhau

Eq. 7-8 chuẩn hoá rồi so từng **tham số** với nghiệm lý tưởng. Cài đúng chữ
(`--topsis_mode literal`) thì client bị phóng đại biên độ chiếm max ở **500/500
chiều** — nên chính nó *là* nghiệm lý tưởng, `C_i` cao nhất, và **không bị loại**.

Nhưng bài lại viết outlier phải gần anti-ideal, và mô tả tiêu chí là *"quality
and consistency"*. Nên mặc định là `--topsis_mode consistency`: giữ nguyên
Eq. 7-8, chỉ thay ma trận tiêu chí bằng ba độ đo nhất quán (khoảng cách tới
trung vị, lệch chuẩn, cosine với trung bình). **Khác biệt này phải ghi trong
báo cáo.**

### 3. Eq. 14 cần một ràng buộc mà bài không nói

Bài cho `|C|` cụm, mỗi cụm Multi-Krum giữ `m = 5`. Nếu cụm **nhỏ hơn hoặc bằng**
`m` thì Multi-Krum giữ hết, tầng 2 thành vô tác dụng. Đo được: `|C|=4` trên 16
update (cụm 4 người) → giữ 16/16, kẻ tấn công lọt qua TOPSIS đi thẳng vào model
global (**bắt 0/1**, trong khi một cụm duy nhất bắt 1/1).

Nên code tự hạ `|C|` xuống `N/(m+1)` và ghi rõ vào log. Bài cũng không cho `|C|`
lẫn thuật toán gom cụm — mặc định `round(sqrt(N))`, cắt theo thứ hạng `C_i`.

Eq. 14 là trung bình **không trọng số** các cụm, không nhân theo số mẫu như
FedAvg. Đo trên dữ liệu giả (20 client, 6 round, không tấn công): gom cụm
0.9135 acc so với một cụm phẳng 0.7524.

### 4. Eq. 15 mặc định TẮT — có lý do đo được

`--personal_coef` mặc định `1.0`, tức đúng hàng *"w/o Personalization"* của
Bảng 6. Vì cá nhân hoá tối ưu độ chính xác **từng xe**, còn thước đo dùng để so
sánh bốn bài là **model global trên tập test chung** — hai thứ khác nhau:

| α | global acc | global macro-F1 | cục bộ acc | cục bộ std |
|---|---|---|---|---|
| 1.0 (tắt) | 0.5114 | 0.4214 | 0.5572 | **0.2847** |
| 0.55 | 0.4951 | 0.4029 | 0.7233 | 0.1709 |
| 0.15 | 0.6238 | 0.5274 | **0.8672** | **0.1556** |

Cá nhân hoá kéo độ chính xác cục bộ 0.557 → 0.867 và **thu hẹp độ lệch giữa các
xe 3.35 lần theo phương sai** — bài nói *"Accuracy variance >4× across
vehicles"* khi bỏ cá nhân hoá, cùng chiều và cùng bậc độ lớn.

Bật bằng `--personal_coef 0.55 --local_val 0.15`. Bài **không công bố** giá trị
α đã dùng; Bảng 3 chỉ cho không gian tìm `{0.15, 0.35, 0.55, 0.75, 0.95}`.

Eq. 15 cần model cục bộ của round trước. Trong chế độ simulation, đối tượng
client bị **tạo lại mỗi round** nên biến thành viên không sống sót — trạng thái
phải ghi ra `<out_dir>/client_state/`. `--restart` sẽ xoá thư mục này.

### 5. Những chỗ bài không định nghĩa

`sigma_Kolmogorov` chỉ được gọi tên, không có công thức — ở đây là SiLU + tổ hợp
Fourier. Số kênh KANConvNet không có trong bài, để `(16, 32)` vì KAN đắt hơn
conv thường ~10×. GA cài thuần numpy thay DEAP (cùng toán tử).

## Sửa code

Repo này là **nguồn gốc của chính nó**. Sửa thẳng ở đây, không có bước build
trung gian nào cả. Sửa repo này không đụng gì tới ba repo kia.

```bash
# sửa file...
push.bat "sua gi do"        # Windows
./push.sh "sua gi do"       # Linux/Mac
```

### Về `common.py` và `model_cnn1d.py`

Hai file này ban đầu giống hệt ở cả 4 repo — bốn bài dùng chung backbone thì so
sánh mới công bằng. Khi bạn sửa riêng ở đây, chúng sẽ lệch dần so với ba repo
kia. **Đó là đánh đổi có chủ đích** để bốn repo độc lập thật sự.

Nhưng nếu đang so sánh kết quả giữa bốn bài thì backbone lệch nhau sẽ làm phép
so sánh mất giá trị. Kiểm tra trước khi kết luận:

```bash
python check_shared.py --against ../VANFED-IDS
```
