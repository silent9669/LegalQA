# Runbook gọn — Kaggle smoke rồi chạy phần có ích

**Cập nhật:** 2026-09-22 · **PLAN ONLY.** Tên file giữ để link cũ hoạt động; không còn chuỗi Kaggle/A100/release gates bắt buộc. Không Colab, không tự gọi Modal. Lượt hiện tại chỉ sửa docs.

## 1. Kaggle smoke “vừa đủ biết code chạy”

Mặc định dùng Kaggle Dual-T4 sẵn có; có thể chạy sequential trên một T4. Không ép DDP hoặc lấp đầy cả hai GPU chỉ để smoke. Precision phải phù hợp T4: không unconditional BF16.

**Giới hạn công việc đề xuất:**32training rows, tối đa8optimizer steps nếu có training change;8QA generation, tối đa128new tokens/câu; khoảng1,000passages cho toy index (gồm positives và distractors). Ghi rõ đây là sample, không phải index/eval production.

1. **Load thật:** tokenizer, đúng model/checkpoint, đúng preprocessing. Không mock/random embeddings khi thiếu dependency/model.
2. **Component đang thay đổi:** nếu train dense thì forward/backward vài bước, loss hữu hạn và weights thực sự đổi; save vào smoke-only folder rồi reload/encode. Nếu chỉ inference/assembly thì bỏ training. Nếu sửa Qwen SFT mới kiểm tra collator completion mask và vài bước SFT.
3. **Mini end-to-end:** retrieval → rerank → evidence → generation → assembly trên8QA. Chạy ít nhất một input dài đúng giới hạn context dự định dùng, nhưng giới hạn output smoke để tiết kiệm. Load/unload sequential nếu cần VRAM; không bắt cả3models đồng trú.
4. **Kiểm tra output:** đúng ID, đủ8câu, không rỗng; JSON parse được; ZIP sample có đúng `submission.json` nếu đường đóng gói thay đổi. Ghi rõ sample, không được dùng làm bài nộp1918câu.
5. **Lưu kết quả ngắn:** versions/model revisions, steps/counts, status, loss/timing thực đo, một vài output và error nếu có. Thiếu metric ghi thiếu; không dùng literal score để PASS.

**Pass là hết lỗi code path này, không phải điểm cao.** Không chạy full corpus encoding, full dev, đủ epochs, full suite hoặc test kill-container trong smoke. Không chấm10câu rồi coi đó là quality gate.

Nếu không fit memory, giảm batch/load tuần tự và ghi cấu hình thực tế. Batch smoke khác batch training lớn là bình thường. OOM không được “khắc phục” bằng cách âm thầm bỏ model cần dùng.

## 2. Sau smoke: thử model, không train lại toàn stack

- Reuse ft_v2 và Qwen adapter đã có theo04; lập baseline nhỏ theo02/05.
- Có matching corpus index thì dùng lại; không có thì encode một lần. Đo riêng thời gian index.
- Chỉ đổi assembly: CPU sweep trên raw output cache; không GPU smoke lại nếu GPU code không đổi.
- Muốn train dense: negatives cached, init/loss/batch rõ ràng, một round trước. Giữ generator/reranker frozen.
- Dùng notebook phục vụ experiment gọn là được; v13 là reference để lấy dense training/serving, không copy nguyên dynamic adapter auto-pick, stale cache hoặc upload cells.

Không yêu cầu R1–R6 hoàn tất hay nguồn code phải có một release manifest nhiều tầng mới được thử. Test gần code thay đổi là đủ để bắt đầu; lỗi thật ảnh hưởng phép thử vẫn cần sửa.

## 3. Nếu chọn GPU lớn hơn cho lượt chính

A100 là lựa chọn thực thi, không phải gate bắt buộc. Chỉ chạy Modal khi có authorization phù hợp; yêu cầu hiện tại không tự bật Modal hay tạo chi phí.

Khi đã được phép chạy một experiment có scope/budget rõ, không hỏi lại từng bước nhỏ trong scope đó. Trước khi chạy chỉ cần biết: platform/GPU, reuse hay train gì, số bước/câu, thời gian/chi phí trần, nơi lưu output. Không cần “production approval packet” dài.

Kaggle smoke không chứng minh BF16/merge/long-generation trên A100. **Kiểm tra vài câu đầu ngay trong job chính** trước khi mở rộng batch là đủ mặc định; không lập thêm một job A100 probe riêng với quy trình chứng nhận. Nếu đổi precision/backend thì first-batch check phải dùng chính cấu hình đó.

Theo dõi actual progress, OOM/Traceback/exit/timeout; không coi im lặng là thành công. Nếu mất tiến độ hoặc sắp hết budget, xem log và lưu phần hoàn thành; không tự kéo dài/restart ngoài quyền đã có. User stop thì dừng, không auto-restart.

## 4. Giữ công đã chạy

- Model/checkpoint và index có identity rõ; không dùng path mutable làm identity duy nhất.
- Cache raw generations key theo prompt, generator weights/tokenizer và decoding settings; lưu theo batch cho lượt dài.
- Đổi retrieval/context → prompt đổi → chỉ reuse cache nếu input generation thực sự giống. Chỉ đổi assembly không cần sinh lại.
- Đếm output mỗi batch, không để `zip()` truncate âm thầm. Lỗi ghi cache phải hiện ra.
- Checkpoint smoke không thay checkpoint tốt; index toy không thay index đầy đủ.
- Upload lỗi chỉ retry upload sau khi được phép; không train lại vì upload lỗi.

Các việc này có thể làm nhỏ trong đường hiện tại. Không cần xây framework resume mới trước experiment8câu.

## 5. Khi thật sự tạo submission

Giữ kiểm tra ngắn vì mất một lần nộp bài cũng tốn thời gian:

- Đúng tập ID/count từ input thực tế; lịch sử private có1,918câu, không hardcode để bỏ qua input mới.
- Tất cả answer là string không rỗng/không header-only; không nhét private gold hoặc câu trả lời thủ công theo private ID.
- JSON không duplicate keys; ZIP đúng một member `submission.json`; bytes JSON trong ZIP đúng bản đã kiểm tra. Lưu hash.
- Xem nhanh vài output dài/ngắn về citation, số tiền/thời hạn/phủ định; không chỉ kiểm schema.
- Lưu recipe/model identities và output thật. Không tạo train.log/trainer_state giả cho run chỉ reuse.
- Upload HF và submit Codabench chỉ khi được phép. Báo rõ compute xong nhưng upload fail nếu xảy ra.

Sau khi có receipt, lưu submission ID + ZIP hash + official scores/metadata. `elapsedTime` của scorer không phải inference time.

## 6. Kết thúc lượt: báo ngắn và thật

1. Đã đổi gì; paired sample thắng/thua/không rõ, cỡ mẫu và exposure.
2. Phút startup/mining/train/index/eval/inference thực tế, không chỉ optimizer time.
3. Model/index/output dùng được ở đâu; phần nào chưa chạy hoặc lỗi.
4. Official score nếu có; nếu chưa có thì không ghi “đạt>0.60”.

Full regression suite, full-dev, bootstrap hoặc independent holdout là **tùy chọn tăng độ tin cậy**, không trở lại thành điều kiện ngầm để được thử nhanh.
