# Sửa đúng chỗ — backlog theo experiment

**Cập nhật:** 2026-09-22 · **PLAN ONLY, chưa sửa code.** Đây không còn là chuỗi R1–R6 phải hoàn thành trước mỗi lần thử. Các finding đầy đủ giữ tại [01](01-evidence-and-corrections.md); đường chạy nhẹ tại [06](06-release-and-a100-runbook.md).

## 1. Chỉ giải quyết lỗi chạm vào đường đang chạy

| Khi làm việc này | Phần cần đúng | Code đã đối chiếu / kiểm tra nhỏ đề xuất |
|---|---|---|
| Load encoder HF/local | Model thật, segmentation rõ ràng, masked mean + L2, max256, đúng checkpoint | `src/common/dense.py`; 4 chuỗi tiếng Việt, encode trước/sau reload, vector hữu hạn; thiếu dependency/model phải lỗi rõ thay vì random vectors |
| Dùng index có sẵn | Corpus content/order, encoder weights, preprocessing, dimension tương thích | 8–16 passage encode lại đối chiếu rows tương ứng; không reuse chỉ vì đều768 chiều |
| Đổi assembly | Cùng prose/evidence, không trộn output từ recipe khác | `src/task2/candidates.py`, `src/task2/evidence_packer.py`; vài fixture về khoản luật, số tiền, phủ định và ngân sách chars |
| So sánh answer metric | Prediction dùng recipe thực tế, scorer có reference đúng | `src/task2/evaluation.py`; kiểm tra không chỉ chấm `stitched_extract` nếu final dùng `dual_assembled`; không bỏ `dual_assembled` khỏi nhóm phụ thuộc generator |
| Chạy smoke cũ | Không in metric giả | `scripts/run_gpu_gate.py`: literals0.482/0.518, timing fallback1.42 và memory key mismatch phải bỏ khỏi báo cáo nếu dùng đường này; có thể dùng smoke notebook gọn riêng thay vì refactor toàn gate framework |
| Train dense mới | Labels đúng, negative sạch, batch/loss thực tế được ghi | Training path của v13 làm reference; vài bước finite loss, weights đổi, save/reload. Không tự dùng trainer reranker hay Qwen thay dense trainer |
| Chỉ khi train Qwen | Completion mask và hyperparameters thực sự áp dụng | `src/task2/generation/trainer.py`: kiểm tra batch từ collator, không bỏ qua mask failure; thay hardcoded warmup_steps=1 nếu recipe yêu cầu warmup ratio |
| Chạy inference dài | Cache nhận diện đúng weights/prompt/decoding và lưu từng batch | `src/task2/predict.py`: không chỉ hash adapter path; kiểm tra số output đúng số input; không nuốt lỗi ghi cache |

Bảng này là scope chọn việc, không yêu cầu làm mọi hàng. Ví dụ: chỉ thử article6000 trên generations đã lưu thì không cần sửa trainer, launcher và uploader.

## 2. Ba sai sót dễ làm mất lợi ích của encoder đã train

1. **Local path làm mất segmentation.** Code hiện suy ra preprocessing từ chữ `dek21` trong model path. Encoder export đặt tên `encoder_ft_v2` vẫn cần đúng preprocessing của lúc train; đừng dựa vào tên folder.
2. **Raw Transformer export bị nạp sai representation.** Dựng rõ Transformer → attention-masked mean pooling → L2 normalize; không giả định export có đầy đủ SentenceTransformer modules. Pooling/normalization/query preprocessing phải đồng nhất giữa mining, training và serving.
3. **Index cũ đi cùng weights mới.** Đổi encoder phải encode lại corpus hoặc dùng index có cùng identity. Không chỉ sửa config model name rồi giữ vectors cũ.

Đây là ưu tiên trước experiment dense vì sai một trong ba thì so sánh không còn đo model muốn thử. Không cần đợi hoàn tất provenance framework mới kiểm tra được ba điểm này.

## 3. Tái sử dụng, không retrain vì lỗi phụ

- Reuse checkpoint hoàn chỉnh đã load được; metadata ghi đúng nguồn. “Load để inference” khác “resume optimizer/scheduler/RNG”.
- Nếu thử continuation, ghi parent checkpoint và actual learning rate; thiếu optimizer state không ngăn fine-tune weights nhưng không được gọi là exact resume.
- Lỗi upload sau khi compute xong: sửa/retry upload nếu được phép; không chạy lại training.
- Cache generation lưu theo batch cho job dài; retry chỉ phần thiếu. Không overwrite checkpoint lịch sử.
- Smoke checkpoint nằm folder riêng, không bao giờ tự chọn theo “folder nông nhất” rồi dùng thay model production.

## 4. Việc có thể để sau

Full fault-injection, exhaustive release manifest schemas, toàn bộ suite không liên quan, tự động orchestration nhiều stage, audit lockbox nhiều tầng. Chỉ kéo việc vào scope khi lỗi thực tế hoặc đường chạy cần nó.

Nếu chọn Modal sau này, kiểm tra launcher gọi `python3 -m modal` bằng argv đúng; không tự fallback Kaggle gate sang Modal T4. Hiện tại chưa có lệnh launch trong tài liệu này.

## 5. Cách verify vừa đủ khi triển khai

Viết/chạy test gần phần thay đổi: encoder reload/index parity nếu sửa dense; reference fixture nếu sửa assembly; collator batch nếu sửa SFT. Sau đó Kaggle smoke theo06 cho đường GPU đã đổi. Mock unit test không chứng minh GPU chạy, nhưng không cần full GPU train chỉ để kiểm tra một parser.

Không có test hay GPU smoke mới nào đã chạy trong lượt viết tài liệu này. Không coi các task trong bảng là đã sửa chỉ vì đã nêu giải pháp.
