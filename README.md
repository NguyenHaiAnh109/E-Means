# e-means bằng NumPy

Bản triển khai nghiên cứu thuật toán e-means dựa trên [arXiv:2611.09501v1](https://arxiv.org/html/2611.09501v1), cụ thể là §3.1–3.3, Algorithm 1, Bảng 2 và Phụ lục H.

Thuật toán nằm trong `emeans.py`. NumPy xử lý dữ liệu và trạng thái huấn luyện. PyTorch là tùy chọn cho bước tính khoảng cách. Scikit-learn được dùng trong benchmark để sinh dữ liệu, tính chỉ số đánh giá và chạy hai mô hình đối chứng.

## Chạy thử

Cần Python 3.10 trở lên. Chạy các lệnh sau từ thư mục chứa `emeans.py`.

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python benchmark.py --output results
```

Ví dụ sử dụng với một mảng NumPy.

```python
import numpy as np

from emeans import EMeans

rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal(loc=-3.0, scale=0.6, size=(200, 2)),
    rng.normal(loc=3.0, scale=0.6, size=(200, 2)),
])

model = EMeans(
    n_clusters=2,
    batch_size=None,
    max_epochs=10,
    alpha_0=None,
    epsilon=0.1,
    random_state=42,
)

labels = model.fit_predict(X)
centroids = model.cluster_centers_
distances = model.transform(X[:20])
new_labels = model.predict(X[:20])

print("Centroids:", centroids)
print("Inertia:", model.inertia_)
print("Batch size:", model.batch_size_)
print("Initial step:", model.alpha_0_)
print("Batch losses:", model.loss_history_[:5])
```

- `fit(X)` huấn luyện mô hình.
- `predict(X)` trả về chỉ số centroid gần nhất cho mỗi điểm.
- `fit_predict(X)` huấn luyện rồi trả về nhãn theo các centroid cuối cùng.
- `transform(X)` trả về ma trận khoảng cách Euclid có shape `(n_samples, n_clusters)`.
- `batch_size=None` và `alpha_0=None` bật scaling law tương ứng. Có thể truyền giá trị thủ công cho từng tham số.
- Sau khi cài PyTorch, đặt `device="cpu"` hoặc `device="cuda:0"` để tính khoảng cách bằng `torch.cdist`. Sampling và cập nhật EMA vẫn chạy bằng NumPy.

## Quy trình huấn luyện

### 1. Khởi tạo và lấy mini-batch

Mặc định, mô hình lấy đều `k` hàng từ dữ liệu mà không hoàn lại để làm centroid ban đầu. Có thể chọn `init="k-means++"` hoặc truyền một mảng centroid có shape `(k, d)`. `random_state` kiểm soát khởi tạo, xáo trộn và chọn donor.

Mỗi epoch xáo trộn lại dữ liệu và chỉ xử lý các batch đầy đủ. Với `n` điểm, batch size `B` và `T` epoch, ngân sách mặc định là `M = T * floor(n / B)` lượt cập nhật. Khi `n` không chia hết cho `B`, phần dư của lượt xáo trộn đó được bỏ qua. Có thể đặt `total_iterations` để dừng sớm hơn ngân sách này.

### 2. Gán cụm và cập nhật EMA

Mỗi điểm được gán cho centroid có bình phương khoảng cách Euclid nhỏ nhất. Mã tính khoảng cách theo khối và chọn chỉ số centroid nhỏ hơn khi hai khoảng cách bằng nhau.

Với cụm `j`, `n_j` là số điểm được gán trong batch hiện tại và `s_j` là tổng vector của các điểm đó. `N_j` và `S_j` là hai đại lượng tích lũy bằng trung bình trượt mũ, nên `N_j` có thể là số thập phân.

```text
Khởi tạo:
    N_j = 1e-10
    S_j = N_j * c_j_initial

Cập nhật cho mỗi batch:
    N_j = gamma * N_j + (1 - gamma) * n_j
    S_j = gamma * S_j + (1 - gamma) * s_j
    c_j = S_j / N_j
```

Trạng thái cũ nhận trọng số `gamma`, còn batch mới nhận trọng số `1 - gamma`. Đóng góp của một batch tiếp tục giảm qua các lần cập nhật sau. Tỉ số giữa tổng vector có trọng số và số điểm có trọng số cho ra centroid. Nếu `N_j` về 0, mã giữ centroid gần nhất đã xác định để tránh chia cho 0.

### 3. Lịch cosine

```text
alpha(t) = alpha_0 * [epsilon + (1 - epsilon) * (1 + cos(pi * t / M)) / 2]
gamma(t) = 1 - alpha(t)
```

Mã dùng `t = 0, ..., M - 1` theo phương trình (5), nên batch đầu nhận đúng `alpha_0`. Bước lớn lúc đầu giúp centroid thích nghi nhanh. Bước giảm dần giúp hạn chế dao động giữa các batch về cuối. Mức sàn `epsilon * alpha_0` nằm tại `t = M`. Mặc định `epsilon = 0.1`.

### 4. Dead cluster revival

Sau mỗi 5 mini-batch, mô hình kiểm tra các cụm có count EMA thấp hơn 10% count EMA trung bình của tất cả các cụm. Một cụm yếu nhận một nửa count EMA của cụm donor. Hai centroid con được tạo bằng cách nhân xen kẽ các tọa độ của centroid donor với `1 + delta` và `1 - delta`, rồi đảo dấu xen kẽ cho centroid con còn lại. Mã dùng `delta = 2**-7` theo epsilon của BF16 và ghi lại `S_j = N_j * c_j` cho cả hai cụm.

Donor được lấy không hoàn lại, với xác suất tỉ lệ theo count EMA, trong số các cụm sống đủ điều kiện. Phép tách dùng trạng thái EMA nên không cần lấy hai điểm từ batch hiện tại. Nếu thiếu donor, các cụm yếu còn lại đợi lần kiểm tra tiếp theo. Mã bỏ qua donor có centroid là vector 0 vì phép nhân sẽ tạo hai centroid trùng nhau.

### 5. Chốt centroid và ghi loss

Mặc định, `finalization="last_epoch_mean"` lấy trung bình các điểm đã được gán vào từng cụm trong lượt huấn luyện cuối. Cụm không nhận điểm trong lượt đó giữ centroid EMA. Đặt `finalization="ema"` để trả trực tiếp các centroid EMA.

`loss_history_` lưu MSE của từng mini-batch trước khi cập nhật. `labels_` và `inertia_` được tính lại trên toàn bộ tập train bằng các centroid cuối cùng.

## Scaling laws

Với `n` điểm train, `k` cụm và ngân sách `T` epoch, mã dùng hệ số trong Bảng 2 của paper.

```text
B_raw = b_c * k**0.13 * n**0.88 * T**0.81

b_c = 0.035   nếu n <= 2_000_000
b_c = 0.0044  nếu n >  2_000_000

alpha_0 = min(0.824 * k**(-0.46) * B**0.90 * n**(-0.45) * T**(-0.11), 1)
```

`B_raw` được làm tròn đến lũy thừa hai gần nhất theo khoảng cách log, rồi giới hạn trong `[1, n]`. Công thức `alpha_0` dùng batch size `B` sau khi đã chọn. Nếu đặt `total_iterations` và để mô hình tự chọn batch size, công thức batch dùng `max_epochs`, còn công thức `alpha_0` dùng ngân sách hiệu dụng `total_iterations / floor(n / B)`.

Các hệ số được fit trên bộ DEEP. Với dữ liệu nhỏ và ít chiều như blobs, có thể dùng tập validation để chọn `B` và `alpha_0` phù hợp hơn.

## Benchmark

`benchmark.py` sinh 6.000 điểm 2D bằng `make_blobs`, với 4 cụm gốc, 4.800 điểm train và 1.200 điểm test. Benchmark thử `k = 3, 4, 6` qua ba seed. Trong mỗi cấu hình, cả ba mô hình dùng cùng centroid khởi tạo và mức trần 10 epoch. Hai phương pháp mini-batch dùng cùng batch size. Cách lấy mẫu và điều kiện dừng có thể khác nhau giữa các mô hình.

| Chỉ số | Cách tính |
| --- | --- |
| Inertia train | Tổng bình phương khoảng cách từ các điểm train đến centroid gần nhất |
| MSE test | Trung bình bình phương khoảng cách Euclid trên mỗi điểm test |
| ARI test | Mức độ khớp giữa nhãn dự đoán và nhãn blobs gốc |
| Silhouette test | Mức độ gắn kết trong cụm và tách biệt giữa các cụm |
| Training time | Thời gian gọi `fit` trên một luồng CPU |

Kết quả trung vị tại `k = 4`.

| Mô hình | Inertia train | MSE test | ARI test | Silhouette test | Fit (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| e-means | 10283.05 | 2.0521 | 0.9628 | 0.6582 | 13.00 |
| KMeans | 10282.53 | 2.0523 | 0.9628 | 0.6582 | 1.86 |
| MiniBatchKMeans | 10286.85 | 2.0551 | 0.9628 | 0.6582 | 7.89 |

Lloyd KMeans có thời gian fit trung vị thấp nhất trên bộ blobs 2D này. Số liệu đầy đủ và biểu đồ được ghi vào `results/`.

- `metrics_by_seed.csv` chứa kết quả của 27 lượt chạy.
- `metrics_median.csv` chứa trung vị theo mô hình và số cụm.
- `dataset.png` vẽ dữ liệu gốc và tập test.
- `clusters.png` vẽ các cụm dự đoán trên tập test cùng centroid.
- `training.png` vẽ loss theo lượt cập nhật và so sánh thời gian huấn luyện.

## So sánh thuật toán

| Thuật toán | Cách cập nhật centroid |
| --- | --- |
| Lloyd KMeans | Gán toàn bộ dữ liệu mỗi vòng rồi lấy trung bình của từng cụm |
| MiniBatchKMeans | Gán theo mini-batch và cập nhật trung bình tích lũy của từng cụm |
| e-means | Cập nhật hai EMA theo lịch cosine, phục hồi cụm yếu bằng cách tách donor và có tùy chọn chốt centroid từ lượt cuối |

## Kiểm thử và giới hạn

Đã chạy 11 kiểm thử NumPy, gồm EMA đối chiếu với phép tính tay, revival khi thiếu donor, dữ liệu có NaN hoặc Inf, mảng `memmap` và khả năng tái lập. Kiểm thử đối chiếu PyTorch được bỏ qua vì môi trường benchmark thiếu PyTorch.

Bản triển khai nhận mảng dense và có thể đọc dữ liệu qua `np.load(..., mmap_mode="r")`. Tính khoảng cách theo khối giúp giảm bộ nhớ tạm. Chi phí assignment cho một batch vẫn là `O(B * k * d)`, xáo trộn giữ `O(n)` chỉ số và `transform` cấp phát kết quả có shape `(n, k)`. Đầu vào có NaN hoặc Inf sẽ báo lỗi. Với dữ liệu có giá trị đủ lớn để làm tràn phép bình phương, cần đổi thang đo trước khi huấn luyện.

Phạm vi kiểm chứng gồm các công thức và benchmark CPU trên blobs. Kernel GPU chuyên biệt, assignment gần đúng bằng IVF, huấn luyện phân tán và finalization slow-EMA nằm ngoài phạm vi của bản triển khai này.