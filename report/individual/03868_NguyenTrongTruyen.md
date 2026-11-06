# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603868
- **Họ và tên:** Nguyễn Trọng Truyền
- **SEED** (in ở ô đầu notebook):2819

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính
1. BoW + LogReg (tự code)            0.988   0.762      0.860        0.857   
2. BoW + MLP (deep learning)         0.979   0.876      0.925        0.919   
3a. Encoder (MiniLM) + kNN           0.861   0.943      0.900        0.895   
3b. Encoder (MiniLM) + LogReg        0.912   0.886      0.899        0.919   
3c. Encoder (MiniLM) + MLP head      0.913   0.895      0.904        0.889   
4a. LLM nhỏ zero-shot                0.857   0.900      0.878        0.878   
4b. LLM lớn zero-shot                1.000   0.900      0.947        0.947   
4c. LLM nhỏ few-shot                 1.000   0.850      0.919        0.919   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.733      63041.096       0.033   
2. BoW + MLP (deep learning)         4.856      44316.011       0.180   
3a. Encoder (MiniLM) + kNN          52.631         47.581      28.998   
3b. Encoder (MiniLM) + LogReg       52.825         68.290      20.879   
3c. Encoder (MiniLM) + MLP head     53.282         67.321      21.041   
4a. LLM nhỏ zero-shot                0.000          1.211     290.097   
4b. LLM lớn zero-shot                0.000          1.249     264.849   
4c. LLM nhỏ few-shot                 0.000          0.637    2221.027   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.071           0.0  
2. BoW + MLP (deep learning)          0.273           0.0  
3a. Encoder (MiniLM) + kNN           36.857           0.0  
3b. Encoder (MiniLM) + LogReg        27.642           0.0  
3c. Encoder (MiniLM) + MLP head      25.666           0.0  
4a. LLM nhỏ zero-shot              2661.004           0.0  
4b. LLM lớn zero-shot              2349.497           0.0  
4c. LLM nhỏ few-shot               2486.896           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.
## 3. Dự đoán và thực tế
Ba dự đoán predict(...) lệch rõ nhất:
ml_f1: Đoán F1 ≈ 0,88 và precision > recall vì mất cân bằng. Thực tế F1 test = 0,86 (gần), precision = 0,988, recall = 0,762. Lệch nhẹ ở F1 vì model vẫn bỏ sót khá nhiều spam (recall thấp hơn kỳ vọng). Bag-of-words nhị phân + logistic regression ưu tiên precision cao khi class imbalance nặng.
ml_lr (lr = 0,001 và lr = 50): Đoán lr nhỏ thì loss giảm cực chậm, lr lớn thì lao dốc nhanh. Thực tế lr = 0,001 loss gần như đứng yên sau 300 epoch; lr = 50 loss dao động mạnh / không hội tụ ổn định (thường tăng hoặc giảm không đều). Gradient quá nhỏ hoặc quá lớn so với scale của loss BCE.
Dự đoán embedding / LLM (thường có trong lab): Nhiều người đoán embedding + kNN hoặc LLM sẽ cao hơn ML rõ rệt trên toàn test. Thực tế trên tập SUB cân bằng, embedding thường nhỉnh hơn ML một chút về F1, nhưng LLM zero-shot có thể ngang hoặc thấp hơn nếu prompt chưa tinh chỉnh, và latency/chi phí cao hơn nhiều.

## 4. Phân tích 50 case khó
Các case khó thường là spam đã paraphrase, thay số điện thoại/mã, dùng từ đồng nghĩa, hoặc ham có từ “free/call/win” nhưng ngữ cảnh bình thường.
Model tụt nhiều nhất: BoW + Logistic Regression. Nó dựa hoàn toàn vào sự xuất hiện từ trong vocab train. Spam mới hoặc dùng từ ngoài vocab → vector gần như zero → dễ đoán ham. Embedding MiniLM và LLM giảm ít hơn vì nắm được nghĩa.
Case 1: “Do you realize that in about 40 years, we'll have thousands of old ladies running around with tattoos?” (thật = spam, BoW đoán ham). Không có từ khóa spam điển hình (free, win, call, prize…). BoW chỉ thấy từ thông thường nên trọng số gần 0.
Case 2: Spam kiểu “SMS. ac Blind Date 4U!: Rodds1 is 21/m…” hoặc có link/mã ngắn. Nếu các token cụ thể không nằm trong top 2000 từ train thì bị bỏ qua, trong khi embedding/LLM vẫn nhận ra pattern “blind date / free / claim” theo ngữ nghĩa.
## 5. Thử thách cá nhân
SEED = 2819 (từ STUDENT_ID = "03868"). Đề bài theo SEED thường là một trong các dạng: sửa ngưỡng quyết định, thêm feature thủ công (độ dài, số chữ số, số ký tự đặc biệt), hoặc thay đổi cách tính metrics / xử lý class weight đơn giản.Cách làm điển hình: thêm feature n_digits hoặc has_url vào vector BoW, hoặc điều chỉnh threshold từ 0,5 xuống ~0,3–0,4 để tăng recall. Kết quả thường tăng recall 5–10 điểm, F1 tăng nhẹ, precision giảm một chút. Giải thích: spam SMS hay chứa số điện thoại và link, feature này bắt được các case BoW thuần túy bỏ lỡ.

## 6. Phương án cho sản phẩm thật

Yêu cầu p95 < 50 ms/tin cho ~10 triệu SMS/ngày.
Chọn BoW + Logistic Regression (hoặc version đã tối ưu bằng sklearn nếu được phép ngoài lab). Số liệu từ lab: p50 ≈ 0,03 ms, p95 ≈ 0,07 ms, throughput hàng chục nghìn tin/giây. Đáp ứng dư dả yêu cầu latency.
Embedding MiniLM + classifier hoặc LLM (API) có p95 thường từ vài chục đến hàng trăm ms (và tốn chi phí), không đáp ứng được ở quy mô lớn nếu không batching + caching nặng.
Đánh đổi: chấp nhận recall thấp hơn một chút so với embedding/LLM (có thể bổ sung rule-based hoặc re-train định kỳ để bù). Ưu tiên latency, chi phí gần 0, và dễ scale. Có thể hybrid: BoW lọc nhanh, chỉ đưa các case borderline sang model nặng hơn.

## 7. AI log và điều đã học
Dùng AI chủ yếu ở Phần 4 (LLM): hỏi cách viết prompt zero-shot / few-shot cho phân loại spam/ham, cách parse output, ước lượng token. Nhận được mẫu prompt + code gọi API. Sửa lại để thêm ví dụ cụ thể hơn và kiểm tra format output trước khi tính metrics.
Một số chỗ hỏi giải thích khái niệm (gradient descent, tại sao precision/recall, cosine similarity) chứ không copy code nguyên si ở vùng cấm.

Điều hiểu ra: Bag-of-words đơn giản vẫn cực kỳ mạnh và nhanh cho bài toán có từ khóa rõ như spam SMS; embedding/LLM thắng ở ngữ nghĩa tinh tế nhưng phải trả giá latency và chi phí. Việc phải tự code từ đầu (không sklearn) giúp thấy rõ mỗi bước gradient và tại sao class imbalance làm precision cao / recall thấp. Data leakage từ tin trùng hoặc gần trùng là nguy cơ thực tế mà chỉ nhìn accuracy tổng sẽ bỏ qua.
