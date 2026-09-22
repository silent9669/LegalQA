# Kiến trúc và training — học từ v13, giảm việc không cần train

**Cập nhật:** 2026-09-22 · **Đề xuất, chưa implement.** Mục tiêu: điểm cao nhất có thể với vòng thử/training ngắn; không phải thay kiến trúc chỉ để trông mới. Bằng chứng cụ thể: [07](07-hf-v13-findings.md).

## 1. Kiến trúc mặc định

```text
Câu hỏi
  ├─ exact known-QA lookup từ dữ liệu được phép (tách riêng khi đánh giá)
  └─ BM25 + DEk21/PhoBERT fine-tuned → RRF → BGE reranker
       → evidence theo Điều/Khoản → Qwen2.5-3B + adapter có sẵn
       → prose + statutory citation assembly → đáp án
```

Không train generator/reranker ở vòng đầu. Giữ generator và BGE cố định để thấy tác dụng của dense/assembly. Không thêm model lớn hoặc ensemble trước khi xác nhận quy định cuộc thi về tổng tham số và dữ liệu; model-card count không tự chứng minh compliance.

**Khởi điểm từ lịch sử, không khẳng định tối ưu:** dense768 chiều; passage256/query128 model tokens; RRF k60; candidate pool100; rerank→context top8; tối đa2fragment/article. Qwen lịch sử max_seq_len5632/max_new_tokens1400. Dùng config hiện chạy được làm control, không đổi đồng thời toàn bộ các số này chỉ để giống v13.

## 2. Reuse model — ưu tiên đầu tiên

**Dense có bằng chứng:** HF `dangphuc2109/legalqa-qwen2.5-3b-adapter`, revision `a6e4805b5a25dec3f3220f6a0a6506ecc5e00e53`, subfolder `runs/20260920-215402/encoder_ft_v2`.

- Export là RobertaModel/PhoBERT-tokenizer; phục dựng mean pooling + L2 và PyVi preprocessing của notebook.
- Download đúng subfolder/revision; không lấy toàn repo chứa nhiều run nếu không cần.
- Reuse index chỉ khi đúng weights + corpus/text order + preprocessing. Nếu không có index matching thì cần một lần full encode; đây vẫn là chi phí dù không train.
- Không mặc định rút xuống256dimension: Matryoshka cho phép thử dimension nhỏ hơn, nhưng phải đổi cả query/corpus/index và đo retrieval lại. Không phải cách tăng tốc phần Transformer tương ứng3×.

**Generator:** ưu tiên adapter production hiện có làm baseline rõ danh tính. Run `run_d2618710d9d0b6de_20260921_154231/final_adapter` là một ứng viên; dùng cùng revision HF đã pin ở trên nếu lấy từ snapshot này. Có thể so thêm adapter của run0.49 khi exact path/weights đã xác định; không tự chọn một folder qua heuristic.

Không gọi adapter cũ tại `checkpoints/generator/hf_adapter` là “best v13” chỉ vì nó tồn tại: manifest pre-purge ghi dataset10, optimizer3steps, seq256. Chưa chứng minh đó là weights notebook HF thực thi đã load. Khôi phục lineage không được kéo dài vô hạn: chưa rõ thì giữ adapter production đã biết và tiến hành thử dense.

BF16 + merged adapter hữu ích trên A100 đã hỗ trợ; Kaggle T4 dùng precision/load mode phù hợp (FP16/NF4 tùy component). Smoke T4 không chứng nhận performance A100.

## 3. Recipe dense đã có bằng chứng để tái hiện khi cần

| Thành phần | Recipe thực thi lịch sử |
|---|---|
| Init | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` base, không phải ft_v1 |
| Examples | 4,166 rows; question + một positive + hai negatives |
| Loss | `MatryoshkaLoss(MultipleNegativesRankingLoss)` |
| Dimensions | 768,512,256 |
| Epochs / actual LR | 2 / 2e-5 |
| Batch / warmup / max length | 64 / 0.10 / 256 |
| Sampler | NO_DUPLICATES, drop_last=True, seed42 |
| Mining | Top100 hybrid; reranker kiểm tra tối đa12 candidates/query |
| Loại false negatives | Exclude gold documents; bỏ near-copy Jaccard>0.60; veto nếu score negative≥positive |
| Serving | Segmentation tương thích → masked mean → L2 |

Đây là **reference recipe**, không yêu cầu train lại nếu checkpoint có sẵn. Số4,166 là dữ liệu lịch sử, không ép tập mới phải có cùng số rows. Pin dependencies của smoke phù hợp; lịch sử dùng SentenceTransformers3.4.1, Transformers4.49.0, không tự upgrade cả stack để thử một loss mới.

## 4. Thử train nối một vòng ngắn — giả thuyết mới, không phải gain đã đo

Chỉ làm sau khi reuse baseline chạy đúng và muốn cải thiện retrieval thêm:

1. Init từ ft_v2 xác định rõ; không âm thầm fallback base. Nếu phải dùng base thì đó là run khác, LR2e-5 reference ở trên.
2. Mine lại negative theo **encoder hiện tại** để tập trung lỗi mới. Giữ một positive/two negatives; loại toàn bộ gold docs và probable false negatives như v13. Cache rows sau khi tạo; không re-mine mỗi epoch.
3. Đề xuất **1epoch, LR1e-5, warmup0.10**, giữ loss/dims/maxlen để chỉ đổi continuation treatment. Đây là thông số thử, chưa có score xác nhận. Không cần sweep toàn bộ LR×epochs.
4. Save weights, kiểm tra load/encode rồi tạo index matching. Có thể sàng lọc ban đầu trên **cùng candidate pool cố định** (có gold, negatives, distractors) trước full encode; kết quả chỉ là ranking diagnostic, không gọi global retrieval recall hoặc candidate ceiling.
5. So retrieval và answer trên cùng mẫu. Nếu không có ích, giữ ft_v2 cũ. Không mặc định train vòng3/vòng4 chỉ vì loss giảm.

Nếu chỉ train trên T4 mà batch64 không vừa: ưu tiên physical batch lớn nhất ổn định. Batch nhỏ hơn là recipe khác. `batch8 × grad_accum8` **không tương đương64 in-batch negatives** của MNRL.

CachedMultipleNegativesRankingLoss là phương án tiết kiệm memory khi cần giữ logical batch lớn: encoding mini-batches rồi tính contrastive objective trên batch lớn. **Không mặc định nhanh hơn**; overhead có thể tăng. Chỉ thử khi batch memory là nút thắt; xác minh tương thích Matryoshka/precision/gradient checkpointing trên version đã pin trước khi dùng. Không copy API mới từ tài liệu main vào3.4.1 mà chưa test.

## 5. Thay đổi rẻ nhất về điểm: assembly không gọi lại Qwen

Dùng cùng raw prose + cùng evidence, thử trước:

- Article4000chars — control lịch sử.
- Article6000chars — local log120câu:0.52450 so với0.52170 ở4000. Gain nhỏ, không bảo đảm generalize.
- Complete-clause selection; nếu reuse/fit Ridge v13 thì fit trên train-only, thử cap640/800 **whitespace metric tokens**, không phải Qwen tokens. Đơn vị `vi_tokens` trong notebook là NFC/lowercase/`.split()`.

Không áp cap650–800words cho cả đáp án. Không bật dedup mạnh chỉ vì trông sạch hơn: log đã ghi giảm METEOR. Không bỏ tất cả prose để chỉ dùng câu trích: citation-only260 trong lịch sử thấp rõ rệt.

Giữ Điều/Khoản, số tiền, thời hạn và phủ định. Có thể thử selection theo ngân sách ở ranh giới khoản hoàn chỉnh thay arbitrary cut, nhưng phải đo — không gán sẵn gain. Encoder tốt hơn không tự đảm bảo assembly tốt hơn; kết hợp chỉ những variant đã chạy cùng nhau.

## 6. Nếu dense tốt hơn mà answer chưa tốt hơn

Dùng các câu thua để chọn **một** nhánh kế tiếp, không làm cả bảng:

| Quan sát | Thử nghiệm tiếp theo | Lý do tiết kiệm |
|---|---|---|
| Gold thường không vào pool100 | Pool200 một lần; kiểm tra label mapping và query legal reference | Chưa cần train Qwen; lịch sử pool400 không hơn200 ở một audit |
| Gold có trong pool nhưng rerank sai | So input formatting hoặc article max/top2-mean aggregation; không cộng tất cả chunks thiên vị điều dài | Reuse retrieved pool, không encode corpus lại |
| Luật đúng, prose sai/thiếu | So hai adapter có sẵn trên cùng60câu; hoặc một prompt pair | Chưa cần SFT mới |
| Prose đủ, citation nhiễu | Offline assembly alternatives | Không inference mới |
| Model sinh sai dù evidence tốt và adapter comparison không giúp | Một Qwen SFT ngắn, một LR trajectory, checkpoint epoch1 trước | Training generator trở thành có lý do; không sweep2–3epochs mặc định |

Fresh Qwen SFT nếu thực sự cần: init và training scope rõ, completion-only loss, evidence giống serving, warmup/mask kiểm tra trên collator thật. Chọn checkpoint bằng answer metric, không bằng token accuracy. Không có ETA hay gain SFT mới ở đây.

HyDE, nhiều generator, agentic RAG, selector nhiều tầng, DPO và train mới reranker không nằm đường mặc định. Chúng có thể đáng thử sau nhưng chưa có bằng chứng lợi ích/chi phí tốt hơn các bước trên.

## 7. Giảm tổng thời gian, không chỉ optimizer time

- Cache negative mining theo data/labels, encoder/index, reranker và mining config; đổi identity liên quan thì invalidate. Random fallback vẫn có thể chứa unlabeled positives; ghi count, không gọi denoise hoàn hảo.
- Cache segmentation/tokenization theo text/preprocessing/tokenizer; length bucketing và encode unique passage texts, map về đúng physical rows.
- Giữ BM25/corpus preparation khi không đổi text; chỉ rebuild dense vectors khi identity dense thay đổi.
- Retrieval/assembly-only changes không được kích hoạt Qwen retrain. Assembly-only changes không được kích hoạt generation mới.
- Trong lượt mới, đo startup/download, mining, train, index, answer evaluation và full inference riêng. Lịch sử train1.2phút nhưng retriever cell17.3phút: tối ưu pipeline xung quanh có thể tiết kiệm hơn đổi optimizer.

**Kết luận:** dùng lại ft_v2 → thử assembly → nếu cần một vòng dense continuation. Đây là đường thử có bằng chứng sát dự án nhất, không phải lời hứa chắc chắn đạt điểm tối đa hay vượt0.60.
