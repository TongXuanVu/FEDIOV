# FedIoV — tái hiện trên CICIoV

> Heidari, Rastegar, Khonsari, *FedIoV*, Future Generation Computer Systems 181 (2026)

Tái hiện trên **CICIoV** (31 đặc trưng, 13 lớp) để so sánh được với ba bài còn
lại và với AFSIC-IoV / FedLiTeCAN trên cùng một backbone.

Ba repo anh em: [VANFED-IDS](https://github.com/TongXuanVu/VANFED-IDS) ·
[FEDIOV](https://github.com/TongXuanVu/FEDIOV) · [IOVFD](https://github.com/TongXuanVu/IOVFD) ·
[SDNFL-IDS](https://github.com/TongXuanVu/SDNFL-IDS)

---

## Ý tưởng được tái hiện

KANConvNet (tích chập Kolmogorov–Arnold) + tìm siêu tham số bằng Genetic Algorithm + tổng hợp chống Byzantine bằng Multi-Krum (m=5). `KANLinear` cài theo efficient-kan, `KANConv1d` theo Convolutional-KANs.

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

## Chạy

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
| `ga_history.csv`, `ga_best.json` | quá trình và kết quả GA |

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

**Trạng thái:** **Đã chạy thật và đạt** trên dữ liệu giả. Multi-Krum bắt đúng 1/1 client sign-flip (điểm Krum 1.26e5 so với 0.18 của client thật). Chưa dò siêu tham số, chưa chạy trên CICIoV thật.

## Khác gì so với bài báo

**Bài nói dùng đồng thời secure aggregation và Multi-Krum — hai cơ chế này xung đột.** Krum cần thấy bản rõ từng update để đo khoảng cách, secure aggregation lại che chính các update đó. Ở đây Multi-Krum chạy trên bản rõ, bảo mật đường truyền giao cho TLS của Flower. GA cài thuần numpy thay DEAP (cùng toán tử). Số kênh KANConvNet để nhỏ (16, 32) vì KAN đắt hơn conv thường ~10×.

Bài gốc **không công bố source code**. Mọi con số phải tự đo lại, không kỳ vọng
khớp bảng kết quả trong bài.

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
