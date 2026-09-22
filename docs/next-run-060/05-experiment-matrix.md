# Ma trận thử ngắn — ưu tiên điểm và thời gian thực tế

**Cập nhật:** 2026-09-22 · **PLANNED, chưa có kết quả mới.** Thay thế E0–E5/L1/F1 của bản trước. Không cần đi hết bảng.

## 1. Vòng đầu: không train mới

| ID | Thử gì | Giữ nguyên | Việc phát sinh |
|---|---|---|---|
| F0 | Một baseline chạy được: adapter production đã biết + base dense/current assembly | Sample IDs, scorer, memory policy | Score baseline; reuse index/generation đúng identity nếu đã có |
| F1 | Thay base dense bằng HF ft_v2 load đúng | Cùng generator, BGE, fusion, packing và assembly củaF0 | Index matching nếu chưa có; retrieval200câu; answer60câu khi có tín hiệu |
| F2 | Article4000 vs6000; thêm một complete-clause policy nếu sẵn | Cùng raw prose/evidence của recipe được chọn | CPU assembly/scoring; không Qwen inference mới |

Baseline ở đây là **local paired baseline**, không lấy official0.524640878 làm score local mặc định. Có thể bắt đầu trực tiếp từ ft_v2 và chỉ chạy base comparison khi cần: không bắt reproduce full production lịch sử.

Giữ tất cả output của60câu để F2 không sinh lại. Nếu F1 retrieval không tăng, kiểm tra serving/index/label trước; nếu đã đúng thì có thể bỏ F1 thay vì cố train thêm cho khớp kỳ vọng.

## 2. Vòng hai: chỉ một nhánh có lý do

| ID | Khi nào chọn | So sánh | Chi phí cần nhìn thấy |
|---|---|---|---|
| T1 | Dense còn nhiều lỗi và muốn thử train nhanh | ft_v2 vs continuation1epoch/LR1e-5 với negatives mới | Mining + train + full corpus encode + answer sample, không chỉ train |
| T1b | Cần replay recipe đã đo hoặc init cũ không phù hợp | Base vs base-trained2epochs/LR2e-5 theo v13 | Là phương án thay T1, không bắt buộc chạy cả hai |
| R1 | Correct evidence vào pool nhưng rank kém | Một aggregation hoặc reranker-input pair | Reuse pool; không retrain generator |
| G1 | Retrieval đúng nhưng prose yếu | Current adapter vs một adapter cũ xác định được | Reuse contexts, generation mới60câu |
| G2 | G1/prompt pair không giải quyết lỗi sinh | Một SFT trajectory ngắn, chấm checkpoint epoch1 | Là fallback đắt hơn, không mặc định train lại2epochs |

Không làm Cartesian sweep loss × LR × batch × encoder × prompt × assembly. Mỗi thử nghiệm trả lời một câu hỏi có thể quyết định giữ/bỏ.

## 3. Budget mặc định gọn

- **Smoke:** theo06, một lần cho code path GPU mới; đổi citation chars không cần train smoke lại.
- **Retrieval:**200câu; cosine/ranking diagnostic trên candidate pool cố định được dùng để loại variant kém trước full encode, nhưng không thay thế full-corpus retrieval metric.
- **Answer:**60câu cho comparison; tối đa hai generator adapters trong lượt đầu. Khi chênh lệch nhỏ có thể mở lên120; không tự yêu cầu full-dev.
- **Dense:** thử một continuation round trước; không train nối lặp vô hạn. Có thể dừng ngay ở model có sẵn nếu hiệu quả hơn về điểm/thời gian.
- **Assembly:**4000/6000 và tối đa một complete-clause policy ở vòng đầu. Muốn thử640vs800 metric tokens của Ridge thì dùng cùng output cache.
- **Chi phí:** chưa gán phút/$ cố định cho T4 hay A10040GB. Dùng số đo của môi trường thực tế; báo tổng startup/mining/train/index/eval/inference.

Đây là default tránh lãng phí, không phải cấm mở rộng khi có tín hiệu tốt. Không dùng thời gian tối thiểu làm lý do chọn model thua điểm rõ rệt; nếu gain nhỏ mà chi phí lớn, nêu trade-off thay vì gọi đó là nâng cấp chắc chắn.

## 4. Ghi và quyết định

Mỗi row kết quả chỉ cần:

```text
id | changed_variable | checkpoint/index identity | sample/exposure
METEOR | ROUGE-L | Hit@1/8, MRR nếu có | empty/missing
startup | mining | training | indexing | evaluation | inference seconds
keep/drop/uncertain | lý do
```

- Cùng mẫu, METEOR tốt hơn, xem nhanh câu thua không thấy lỗi nghiêm trọng mới → giữ ứng viên.
- Điểm gần nhau → giữ cách nhanh hơn hoặc mở mẫu nếu đáng; không gọi khác biệt nhỏ là chắc thắng.
- Retrieval tăng nhưng answer giảm → không promote chỉ dựa retrieval/loss; xem packing, reranking, assembly.
- Smoke pass mà metric không tăng vẫn là experiment hợp lệ. Ghi kết quả thua rồi đổi hướng, không chỉnh scorer.
- Sau hai biến thể cùng nhánh không có ích, quay lại best-known recipe thay vì tiếp tục train vô hạn.

Không còn promotion threshold delta0.010, CI>0 hay local>0.60 bắt buộc. Final choice là quyết định thực nghiệm với mức bất định được công bố. Có thể chạy đánh giá rộng hơn một lần nếu cần thêm tự tin, không phải gate cho mọi iteration.

## 5. Điểm nào được gọi là thành công?

- `SMOKE_OK`: code path nhỏ chạy; chưa biết chất lượng.
- `SCREENING_BETTER`: tốt hơn paired sample; kèm exposure status và cỡ mẫu, không phải official.
- `FASTER_SIMILAR_SCREENING`: chất lượng gần nhau ở mẫu đã đo, nhanh hơn ở thời gian đã đo; không kết luận tương đương thống kê.
- `NO_GAIN` / `INCONCLUSIVE`: kết quả trung thực, không phải lỗi quy trình.
- `OFFICIAL_IMPROVED`: receipt mới tốt hơn0.524640878.
- `OFFICIAL_ABOVE_060`: receipt mới thực sự >0.60.

Không cộng gain dự kiến của dense + assembly + SFT thành một điểm dự báo.
