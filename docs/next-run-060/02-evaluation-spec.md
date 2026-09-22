# Đánh giá gọn — đủ để chọn hướng, không kéo dài mỗi lần thử

**Cập nhật:** 2026-09-22 · **Đề xuất, chưa thực thi.** Thay thế yêu cầu full-dev/lockbox/CI bắt buộc của bản trước.

## 1. Ba việc khác nhau

| Việc | Cỡ mẫu mặc định đề xuất | Trả lời câu hỏi gì? |
|---|---|---|
| Kaggle smoke | 8 câu QA; 32 training rows nếu có train; corpus nhỏ | Code/model/save-reload/output có chạy không? |
| Retrieval screening | 200 câu có relevance labels | Encoder hoặc ranking mới có tìm đúng luật hơn không? |
| Answer screening | 60 câu có references; có thể mở rộng lên 120 | Output cuối có METEOR/ROUGE-L tốt hơn với chi phí hợp lý không? |

Smoke **không dùng để chọn model theo điểm**. Cỡ mẫu có thể giảm nếu ít dữ liệu hoặc tăng khi kết quả nhiễu. Không phải chạy cả ba mỗi khi chỉ đổi một hằng số assembly.

## 2. Chọn mẫu một lần, dùng lại

- Dùng QA group/canonical IDs hiện có nếu khả dụng; seed42; lưu danh sách ID. Không tạo bộ chia dữ liệu mới chỉ để đủ thủ tục.
- Các biến thể dùng cùng ID, reference, scorer, memory policy và cách join relevance labels. So sánh paired theo từng câu, không so scalar từ hai tập khác nhau.
- Nếu train mới retriever/selector, loại các câu đánh giá và normalized-question duplicates khỏi training/mining rows. Luật công khai vẫn được nằm trong retrieval corpus; câu hỏi/đáp án evaluation không được dùng làm training supervision.
- Không trộn reference vào prompt, chọn passage bằng gold trong evaluation, hoặc thêm đáp án held-out vào QA memory. Các gold passages chỉ dùng cho metric/oracle diagnosis, không dùng làm input cho hệ thống đang chấm.
- Reused checkpoint có exposure chưa biết vẫn được thử và dùng để chọn hướng thực dụng. Ghi `diagnostic_unknown_exposure`; checkpoint đã thấy evaluation ghi `diagnostic_overlap`. **Không bắt train lại cả hệ thống để được thử**, nhưng không gọi những điểm này là held-out hay dự báo official.
- Chỉ ghi `held_out_verified` khi thành phần học được và QA memory đều không thấy các nhóm đó. Chia lại ID sau khi load checkpoint không xóa exposure cũ.

Ưu tiên giữ mẫu đánh giá cũ của v13 nếu khôi phục được IDs và mapping. Không khôi phục được thì dùng mẫu cố định mới và công bố không so trực tiếp scalar với log cũ.

## 3. Đo đúng thứ đang đổi

**Retriever:** base vs ft_v2, hoặc ft_v2 vs continuation; giữ generator, reranker, context policy và assembly. Đánh giá với toàn corpus đúng encoder khi chọn chất lượng cuối; subset corpus của smoke chỉ chứng minh code chạy. Candidate Hit@100 cho biết passage đúng có vào pool; reranked Hit@1/8 và MRR cho biết xếp hạng. Ghi denominator và label không resolve được; không tự biến chúng thành negative.

**Assembly:** cache raw prose và evidence đúng recipe một lần. Quét article4000, article6000 và complete-clause alternative bằng CPU. Không gọi Qwen lại chỉ vì đổi citation budget. Nếu đổi nội dung context/prompt hoặc generator, raw generation cache cũ không còn là output của recipe mới.

**Generator:** chỉ so sánh adapter khi retrieval/assembly đã cố định; bắt đầu với tối đa hai adapter xác định được lineage. Loss/accuracy lúc train không thay cho answer metric.

Scorer giữ đúng BTC (METEOR whitespace tokenization và ROUGE-L); không sửa công thức để làm tăng score. Local score không so bằng phép trừ với official score rồi kết luận cải thiện causal.

## 4. Quy tắc chọn nhẹ

1. Sai model/index, nonfinite, thiếu output hoặc reference lọt vào prediction: sửa đúng lỗi, không dùng metric đó để chọn.
2. METEOR tăng trên cùng mẫu và không xuất hiện lỗi pháp lý nghiêm trọng mới trong các câu xem tay: giữ làm ứng viên. Xem thêm ROUGE-L và từng câu thua nhiều; không tự loại chỉ vì không đạt delta0.010.
3. Điểm gần nhau: ưu tiên cách nhanh/đơn giản hơn, hoặc mở rộng từ60 lên120 câu nếu quyết định đáng tốn thêm inference. Ghi “chưa phân biệt rõ”, không tuyên bố thắng chắc.
4. Xem khoảng10 câu: vài câu tăng mạnh, giảm mạnh và câu ngẫu nhiên. Đọc đúng luật, số tiền, thời hạn và phủ định. Đây là kiểm tra lỗi nhanh, không phải ước lượng tỷ lệ lỗi của toàn bộ hệ thống.
5. Hai thử nghiệm có thêm chi phí mà không có tín hiệu đáng giữ: dừng nhánh đó; không train nối vô hạn vì loss vẫn giảm.

Screening lặp lại trên cùng mẫu có thể overfit lựa chọn. Nếu cần độ tin cậy cao hơn cho ứng viên cuối, **tùy chọn** đánh giá thêm một tập chưa dùng để chọn, full-dev hoặc paired bootstrap. Không còn là gate mặc định; cũng không đổi nhãn diagnostic thành clean chỉ nhờ tăng cỡ mẫu.

## 5. Bản ghi tối thiểu

Một thư mục/run là đủ; không bắt buộc hệ thống manifest nhiều tầng:

```text
recipe.json       # model revision/subfolder; index/preprocessing; ID mẫu; changed variable; exposure
predictions.jsonl # per-ID raw/final answer + nguồn evidence, đủ tính lại score
summary.json      # n, METEOR, ROUGE-L, retrieval metrics nếu có, thời gian theo stage
```

Nếu repo đã có format tương đương thì dùng lại. Log optimizer giữ riêng khi thực sự train. Missing metric là `null` + lý do, không điền hằng số/fallback. Ghi cả biến thể thua, không chỉ best score.

**Official target:** chỉ receipt mới có METEOR >0.60 mới xác nhận đạt mục tiêu. Điểm nhỏ-sample tốt giúp chọn việc tiếp theo, không bảo đảm private score tăng.
