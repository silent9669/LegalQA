# LegalQA next run — ưu tiên điểm cao và vòng thử nhanh

**Cập nhật:** 2026-09-22 · **Trạng thái:** RESEARCH + PLAN ONLY.  
**Phạm vi:** sửa tài liệu trong folder này; chưa sửa runtime/config, chưa train, chưa chạy Kaggle/Modal, chưa upload hoặc nộp bài.

## Quyết định mới

**Tái dùng generator, khai thác đúng retriever PhoBERT/DEk21 đã fine-tune, thử assembly trên output cache; chỉ train thêm khi phép so sánh nhỏ cho thấy đáng làm.** Không mặc định train lại Qwen hoặc toàn bộ pipeline.

Đây là hướng thay thế quy trình dài ngày 2026-09-21, theo yêu cầu mới của người dùng: điểm cao nhất có thể, thời gian training thấp, Kaggle chỉ kiểm tra vừa đủ code hoạt động. Không có recipe nào hiện được chứng minh chắc chắn vượt **0.60**.

- Điểm official đã có file xác nhận: **METEOR 0.524640878**, **ROUGE-L 0.402908777**.
- Notebook v13.2 thực thi trên HF train **dense retriever**, sau đó dùng **Qwen + adapter có sẵn**. Không phải base Qwen zero-shot và không train Qwen trong notebook đó.
- A/B lịch sử: dense MRR **0.2900 → 0.34075** trên 400 câu; reranked Hit@1 **0.3467 → 0.3667** trên 150 câu. Đây là retrieval metrics, không phải official METEOR.
- Train encoder ghi **1.2 phút**, nhưng cả cell khoảng **17.3 phút** trên **A100 80GB PCIe**, gồm mining và encode corpus. Không phải ETA trên Kaggle T4.
- Tên `ft_v2` không chứng minh train nối: log và metadata của lượt này ghi **khởi tạo từ base, LR 2e-5**.
- Cắt ngắn đáp án/dedup mạnh đã làm giảm local METEOR trong log. **Bỏ mục tiêu ép 650–800 từ.** Chọn assembly bằng so sánh thực tế.

## Luồng mặc định — không phải chuỗi release gates

1. **Reuse trước:** nạp đúng `encoder_ft_v2` + một Qwen adapter đã xác định rõ; đối chiếu base dense nếu cần. Không train mới để có baseline.
2. **Một Kaggle smoke ngắn:** load thật → vài bước train *chỉ nếu đang đổi training* → save/reload → retrieval/generation vài câu → kiểm tra JSON. Không chạy đủ corpus hoặc đủ epochs trong smoke.
3. **So sánh nhỏ, cố định:** khoảng 200 câu retrieval và 60 câu answer; reuse raw generations khi chỉ đổi assembly. Mẫu nhỏ để chọn hướng, không để chứng minh điểm private.
4. **Train nhỏ nếu có ích:** tối đa một thử nghiệm dense continuation trước; hoặc replay recipe từ base nếu cần đối chứng. Không bắt buộc làm cả hai.
5. **Giữ biến thể tốt hơn hoặc nhanh hơn ở chất lượng tương đương.** Mở rộng lên khoảng 120 câu answer khi chênh lệch nhỏ/khó hiểu, không phải full-dev bắt buộc.

Các kích thước trên là mặc định đề xuất, không phải số đo hay rào cản cứng. Muốn dùng notebook kiểu v13 để thử nhanh được; không cần dựng lại một hệ thống orchestration mới.

## Đã bỏ khỏi đường thử nhanh

- Hoàn thành toàn bộ R1–R6 trước khi được thử model.
- Bắt train lại generator/reranker dù không thay đổi chúng.
- Full test suite, fault injection toàn hệ thống, full-dev, bootstrap 10,000 lần và lockbox bắt buộc cho từng iteration.
- Ngưỡng promotion cố định `delta ≥ 0.010`, CI phải dương, hay bắt local score >0.60 mới được cân nhắc chạy tiếp.
- Kaggle → A100 probe riêng → production như một chuỗi chứng nhận bắt buộc.
- Nhắc lại approvals cho từng thao tác nhỏ bên trong một experiment đã được cho phép. Modal/chi phí cloud và upload/nộp bài vẫn cần quyền phù hợp; tài liệu này không tự cấp quyền đó.

**Vẫn giữ:** đúng checkpoint/index, không dùng random embeddings thay model thật, không đưa reference evaluation vào prediction, không báo metric giả, không đóng gói thiếu câu. Những kiểm tra này bảo vệ thời gian GPU và độ tin cậy của phép thử, không phải thủ tục release.

## Đọc gì trước?

| Tài liệu | Mục đích |
|---|---|
| [07 — HF và v13](07-hf-v13-findings.md) | Bằng chứng thực thi, recipe retriever, thời gian, giới hạn của lịch sử điểm |
| [04 — Kiến trúc và training](04-quality-implementation-plan.md) | Kiến trúc khuyến nghị, reuse, dense continuation, phương án khác khi cần |
| [05 — Ma trận ngắn](05-experiment-matrix.md) | Thử gì trước; chi phí nào tránh được; khi nào dừng |
| [06 — Kaggle smoke và chạy tiếp](06-release-and-a100-runbook.md) | Smoke đủ dùng, chạy thật không lặp kiểm định dài, submission hợp lệ |
| [02 — Đánh giá gọn](02-evaluation-spec.md) | Cùng câu/cùng scorer; tách diagnostic với held-out; ghi kết quả tối thiểu |
| [03 — Sửa đúng chỗ](03-reliability-implementation-plan.md) | Backlog theo thành phần thực sự dùng, không phải sáu gate bắt buộc |
| [01 — Bằng chứng official](01-evidence-and-corrections.md) | Giữ nguyên receipt và đính chính các suy luận sai trong báo cáo cũ |

**Thứ tự hiệu lực:** README và các tài liệu 02–07 cập nhật ngày 2026-09-22 thay thế quy trình cũ. Các finding F01–F19 trong 01 là ghi nhận kỹ thuật, không phải danh sách phải hoàn tất trước mọi thử nghiệm.

## Việc chưa làm

Chưa có score mới, chưa đo tốc độ trên Kaggle, chưa tải model weight binaries, chưa xác định chắc adapter từng tạo ra benchmark 0.5486. Tài liệu nghiên cứu không đồng nghĩa GPU-ready. Bước triển khai kế tiếp nhỏ nhất là nạp/serve đúng encoder HF và tạo smoke nhẹ, không phải triển khai toàn bộ backlog.
