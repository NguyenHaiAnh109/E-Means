# e-means bằng NumPy

Bản triển khai nghiên cứu dựa trên [arXiv:2611.09501v1](https://arxiv.org/html/2611.09501v1), chủ yếu là §3.1–3.3, Algorithm 1, Bảng 2 và Phụ lục H. Mã thuật toán nằm trong `emeans.py`. Scikit-learn chỉ xuất hiện trong benchmark để sinh dữ liệu, tính metric và chạy các baseline.

## Chạy thử

Dùng Python 3.10 trở lên. Chạy từ thư mục chứa các file sau khi giải nén.

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python benchmark.py --output results
```

```python
import numpy as np

from emeans import EMeans

X = np.load("vectors.npy", mmap_mode="r")  # Shape (n_samples, n_features)
model = EMeans(
    n_clusters=8,
    batch_size=None,
    max_epochs=10,
    alpha_0=None,
    epsilon=0.1,
    random_state=42,
)
labels = model.fit_predict(X)
print(model.cluster_centers_.shape, model.inertia_)
print(model.batch_size_, model.alpha_0_, model.total_iterations_)
print(model.loss_history_[:5])
distances = model.transform(X[:20])
new_labels = model.predict(X[:20])
```

`batch_size=None` và `alpha_0=None` kích hoạt scaling laws. Có thể truyền một trong hai hoặc cả hai. Đặt `init="k-means++"` để dùng cách lấy mẫu theo bình phương khoảng cách, hoặc truyền mảng centroid ban đầu có shape `(k, d)`. Mặc định, e-means lấy đều $k$ hàng không hoàn lại. Các centroid ban đầu có thể trùng tọa độ nếu các hàng trong dữ liệu trùng nhau. Đặt `device="cpu"` hoặc `device="cuda:0"` để dùng `torch.cdist` ở bước tính khoảng cách, sau khi cài PyTorch phù hợp. NumPy quản lý sampling, EMA, finalization và các giá trị trả về trong cả hai chế độ.

## Quy trình training và các quyết định triển khai

| Bước | Theo paper v1 | Quyết định trong mã |
| --- | --- | --- |
| Khởi tạo | Lấy đều các hàng, không hoàn lại. Đặt $N_j=10^{-10}$, $S_j=N_jc_j^{(0)}$. | `random_state` điều khiển sampling, xáo trộn và revival. Có thêm k-means++ và mảng seed cố định. |
| Sampling | Xáo trộn lại dữ liệu từng epoch, với $M=T\lfloor n/B\rfloor$ lượt cập nhật trong §3.1. | Chỉ lấy batch đầy đủ. Khi $n$ không chia hết cho $B$, phần dư của mỗi lần xáo trộn được bỏ qua trong lượt đó. `total_iterations` có thể rút ngắn ngân sách, tối đa $M$. |
| Assignment | Chọn centroid gần nhất theo bình phương khoảng cách Euclid. | Tính trực tiếp theo khối bằng float64. Nếu hòa, lấy chỉ số centroid nhỏ nhất. `transform` trả về khoảng cách Euclid sau khi lấy căn. |
| EMA | Cập nhật đồng thời $N_j,S_j$ rồi tính $c_j=S_j/N_j$. | Khi $N_j=0$, giữ tọa độ centroid gần nhất đã xác định để tránh chia cho 0. |
| Cosine | Dùng phương trình (5), với $t=0,\ldots,M-1$ và mặc định $\varepsilon=0.1$. | Dùng $t=0$ cho batch đầu theo phần giải thích phương trình. Algorithm 1 tăng biến đếm trước dòng schedule. Mã dùng bộ đếm update đã hoàn tất để gọi revival sau batch 5, 10, 15… |
| Revival | Cluster yếu có $N_j<0.1\operatorname{mean}(N)$. Tách donor và chia đôi count EMA theo Phụ lục H. | Chọn donor không hoàn lại từ các cluster sống có count có thể chia đôi và centroid khác vector 0. Khi donor không đủ, các cluster còn lại đợi lần kiểm tra tiếp theo. |
| Finalization | Mặc định lấy trung bình theo các assignment được tạo trong epoch cuối. | Gom count và sum ngay khi xử lý batch của lượt cuối, kể cả lượt bị rút ngắn. Nếu một cluster không có assignment trong lượt cuối, giữ centroid EMA. Tùy chọn `finalization="ema"` trả thẳng các centroid EMA. |
| Chấm điểm | MSE tính theo bình phương chuẩn Euclid trung bình trên mỗi sample. | `loss_history_` ghi MSE batch trước update. `labels_` và `inertia_` được tính lại trên toàn bộ tập train với centroid trả về. |

### EMA và cosine

Trong mini-batch $t$, $n_j$ là số điểm vừa được gán vào cụm $j$, còn $s_j$ là tổng vector của các điểm đó. $N_j$ và $S_j$ là các đại lượng đã được làm mượt qua nhiều batch, nên $N_j$ có thể là số thập phân.

$$
N_j\leftarrow\gamma_t N_j + \alpha_t n_j,
\qquad
S_j\leftarrow\gamma_t S_j + \alpha_t s_j,
\qquad
c_j=\frac{S_j}{N_j}.
$$

Batch hiện tại đóng góp với trọng số $\alpha_t$. Qua mỗi update tiếp theo, đóng góp cũ lại nhân thêm một $\gamma$. Tỉ số giữa tổng vector có trọng số và số điểm có trọng số cho ra centroid. Nếu $n_j>0$, trọng số thực tế của trung bình batch mới trong centroid là $\alpha_t n_j/(\gamma_tN_j^{\mathrm{cũ}}+\alpha_tn_j)$. Vì vậy hai cluster trong cùng batch có thể dịch chuyển với tỉ lệ khác nhau dù dùng chung lịch cosine.

$$
\alpha_t=\alpha_0\left[\varepsilon+(1-\varepsilon)
\frac{1+\cos(\pi t/M)}{2}\right],
\qquad \gamma_t=1-\alpha_t.
$$

Bước lớn lúc đầu giúp centroid thích nghi nhanh với dữ liệu. Bước nhỏ dần giúp giảm dao động giữa các mini-batch ở cuối. Mẫu số giữ nguyên là $M$. Mức sàn $\varepsilon\alpha_0$ nằm tại $t=M$, nên batch cuối thực tế ở $t=M-1$ vẫn có bước cao hơn sàn nếu $\varepsilon<1$.

### Dead cluster revival

Cluster ít được gán điểm đóng góp rất ít vào mục tiêu phân cụm. Mỗi năm mini-batch, mã lấy các donor với xác suất tỉ lệ theo $N_j$ trong tập donor hợp lệ. Hai centroid con được tạo từ centroid cha bằng cách nhân từng tọa độ lần lượt với $(1+\delta,1-\delta,\ldots)$ và $(1-\delta,1+\delta,\ldots)$, với $\delta=2^{-7}$ theo BF16. Mỗi con nhận một nửa count của donor và được ghi lại $S=Nc$. Phép tách diễn ra trên trạng thái EMA nên donor không cần có hai điểm trong mini-batch hiện tại. Với dữ liệu trùng nhau hoặc donor có tọa độ đều bằng 0, số cụm khác nhau đạt được có thể nhỏ hơn $k$.

### Scaling laws

Với $n$ điểm train, $k$ cluster và ngân sách $T$ epoch, dùng hệ số công bố trong Bảng 2.

$$
B_{\mathrm{raw}}=b_c k^{0.13}n^{0.88}T^{0.81},
\qquad
b_c=
\begin{cases}
0.035 & n\le 2\,000\,000 \\
0.0044 & n>2\,000\,000
\end{cases}
$$

Làm tròn $B_{\mathrm{raw}}$ đến lũy thừa hai gần nhất theo khoảng cách log, rồi chặn trong $[1,n]$. Dùng $B$ đã chọn để tính

$$
\alpha_0=\min\left(0.824\,k^{-0.46}B^{0.90}n^{-0.45}T^{-0.11},1\right).
$$

Khi có `total_iterations` và $B$ được chọn tự động, công thức batch dùng `max_epochs` để tránh phụ thuộc vòng. Sau khi biết $B$, công thức step dùng $T=\text{total_iterations}/\lfloor n/B\rfloor$. Các hệ số được fit trên DEEP. Với dữ liệu ít chiều và nhỏ như blobs, giá trị tối ưu của $B,\alpha_0$ vẫn cần xác định bằng validation nếu mục tiêu là chọn cấu hình tốt nhất. Benchmark đi kèm dùng trực tiếp scaling laws để kiểm tra cách chúng hoạt động trên ví dụ 2D.

Mã tham chiếu [Faiss](https://github.com/facebookresearch/faiss/blob/main/contrib/e_means/base.py) có cách làm tròn theo khoảng cách log và các mặc định khác paper, bao gồm hệ số fit chi tiết hơn, giới hạn batch và ngưỡng perturbation. Bản triển khai này dùng các giá trị công bố trong paper v1.

## Benchmark

Dữ liệu có 6.000 điểm 2D và 4 cụm sinh từ `make_blobs`. Chia stratified thành 4.800 train và 1.200 test. Thử $k\in\{3,4,6\}$ qua ba seed, chung centroid khởi tạo trong từng cấu hình và cùng mức trần 10 epoch. MiniBatchKMeans và e-means dùng cùng batch size. Các phương pháp có cách lấy mẫu và điều kiện dừng riêng, nên trần epoch không đồng nghĩa với cùng lượng tính toán. Thời gian chỉ đo lời gọi `fit` trên một luồng CPU. Phần dự đoán test và tính metric nằm ngoài thời gian đo.

- **Inertia train** là tổng bình phương khoảng cách từ từng điểm train tới centroid gần nhất.
- **MSE test** là trung bình bình phương chuẩn Euclid trên từng điểm test, theo cách định nghĩa của paper. Giá trị không chia thêm cho số chiều.
- **ARI test** so assignment với nhãn blobs thật. **Silhouette test** đo độ tách biệt theo khoảng cách Euclid và được ghi NaN nếu số nhãn dự đoán không hợp lệ.
- `results/metrics_by_seed.csv` ghi 27 lượt chạy. `metrics_median.csv` ghi trung vị theo $k$ và mô hình.
- `dataset.png` vẽ dữ liệu gốc và phần test. `clusters.png` vẽ assignment trên tập test cùng centroid. `training.png` vẽ loss theo update và thời gian fit.

Ở $k=4$, ba mô hình có ARI trung vị xấp xỉ 0,963. Khi tăng lên $k=6$, MSE giảm và ARI giảm vì nghiệm chia nhỏ các cụm sinh dữ liệu. Lloyd KMeans có thời gian fit trung vị thấp nhất trên benchmark CPU nhỏ này.

## So sánh thuật toán và giới hạn

- **Lloyd KMeans** gán toàn bộ tập train mỗi vòng rồi thay centroid bằng trung bình của các nhóm vừa gán.
- **MiniBatchKMeans** gán từng mini-batch và cập nhật trung bình tích lũy theo số lượt gán vào cụm. Baseline dùng triển khai scikit-learn và giữ cơ chế reassignment mặc định.
- **e-means** dùng hai EMA với bước theo cosine, chia donor cho cluster yếu và có thể chốt centroid từ assignment của lượt cuối.

Mã ưu tiên assignment Euclid chính xác cho mảng dense. Chi phí một batch là $O(Bkd)$. Chia khối giảm bộ nhớ tạm và hỗ trợ mảng ánh xạ từ đĩa, nhưng phép xáo trộn vẫn giữ $O$n$$ chỉ số và truy cập ngẫu nhiên có thể chậm trên đĩa. `transform` cấp phát mảng kết quả $n\times k$. Đầu vào NaN và Inf bị báo lỗi. Dữ liệu có độ lớn làm tràn số khi tính bình phương cần được đổi thang đo.

Bản này bao phủ quy trình EMA và last-epoch mean trong paper. Các thí nghiệm phân tán, kernel GPU chuyên biệt, IVF, và finalization slow-EMA ở ngoài phạm vi bản triển khai. Backend PyTorch chỉ thay bước tính khoảng cách và có chi phí truyền dữ liệu khi dùng CUDA. Đã chạy 11 test NumPy và 27 lượt benchmark CPU. Test đối chiếu PyTorch được bỏ qua do môi trường chạy thiếu PyTorch. Kết quả benchmark dùng Python 3.12.14, NumPy 2.3.5, scikit-learn 1.8.0 và Matplotlib 3.10.8.
