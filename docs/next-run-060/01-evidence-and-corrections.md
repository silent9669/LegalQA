# Bằng chứng và đính chính

**Cập nhật phạm vi 2026-09-22:** giữ bằng chứng official và findings của lần kiểm tra code trước. Nghiên cứu HF/v13 mới nằm ở [07](07-hf-v13-findings.md); thứ tự thử nhanh theo [README](README.md). Các mã R1–R6/Q1–Q5 bên dưới là cross-reference lịch sử, không còn là chuỗi task bắt buộc.

## 1. Nguồn và mức tin cậy

Nguồn ban đầu: `/Users/phucdang/Downloads/Bao_Cao_Chay_Modal_A100_LegalQA_Run_20260921.md`. Báo cáo này có dữ kiện đúng xen lẫn suy luận không hợp lệ; không dùng nguyên trạng làm specification.

| Nguồn | Dữ kiện | Giới hạn |
|---|---|---|
| [scores.json](evidence/scores.json) | `meteor=0.524640878`, `rouge=0.402908777` | Không chứa submission ID hay hash ZIP |
| [scoring-metadata.txt](evidence/scoring-metadata.txt) | `exitCode=0`, `elapsedTime=705.114177942276` | Thời gian scorer, không phải inference |
| [outer-metadata.txt](evidence/outer-metadata.txt) | Cả bốn trường là `null` | Không ghi đè metadata có giá trị; không suy ra run thất bại |
| Ảnh Codabench đã được người dùng cung cấp | Submission `937816`, Finished, Score làm tròn 0.5246; giờ hiển thị 2026-09-21 22:44 | Ảnh không xác nhận timezone; chưa thêm bản ảnh vào folder này |
| `modal_full_a100.log` | Resumed execution `wall_seconds=12203`; generated 1837/1837; HF upload commit | File local bị gitignore, không phải artifact tự tái tạo từ git |
| ZIP local trong `artifacts/submissions/d2618710d9d0b6de/` | 1,918 predictions, duy nhất `submission.json`, empty=0 | Valid format không chứng minh pháp lý đúng |
| `configs/task2/algorithm.yaml` | Dense base DEk21, 2 epochs, warmup_ratio 0.05 | Phải kiểm tra propagation, không mặc định runtime dùng tất cả giá trị |
| `git show 5f75cfa:notebooks/DSC2026_LegalQA_Pipeline_v13.ipynb` | Nhánh `USE_TRAINED_ADAPTER=False` vẫn load pre-trained PEFT adapter | Không phải base-model zero-shot; exact adapter revision còn cần xác minh |

**Liên kết receipt:** người dùng đưa scores/metadata trong ngữ cảnh run 937816; kết hợp ảnh cho thấy METEOR làm tròn khớp. Không có submission ID/hash trong JSON nên ghi `association=user_context`, không gọi đó là liên kết mật mã đã xác minh.

### SHA-256 của bằng chứng nguyên byte

```text
scores.json
3f268d907f95b1d722fd47711f0b49786138149b83c8425e056abf0be5665e18
scoring-metadata.txt
2b49a7b1478f6e0d495d337a9605edaf8b3d4308e79dc070da912b3ed7456a0f
outer-metadata.txt
a62d27d67512ac74cce9301f48e7678bbebaec720b840bfbba4de883c1e904c6
submission.json.zip (local, không copy vào docs)
91cd575ab8f6041bdbc34bf34afff9e023ca58aa4cdf4fc793fc053ddc85d21c
```

## 2. Kết quả phải ghi nhận

| Chỉ số | Giá trị | Tình trạng |
|---|---:|---|
| Official METEOR | **0.524640878** | Đọc trực tiếp file BTC |
| Official ROUGE-L | **0.402908777** | Đọc trực tiếp file BTC |
| So với 0.4900 | +0.034640878 | Mốc cũ được báo trước, độ chính xác tới 4 chữ số |
| So với 0.5486 | −0.023959122 | Mốc benchmark do người dùng cung cấp, không có receipt trong folder này |
| Khoảng cách tới 0.60 | 0.075359122 | Phải tăng lớn hơn giá trị này để đạt >0.60 |
| Predictions | 1,918 | Đếm lại từ ZIP local |
| Words mean / median | 1065.279979 / 992 | `len(answer.split())`, không phải Qwen token count |
| Words min / max / empty | 87 / 2301 / 0 | Đếm lại; không suy ra đầy đủ hoặc đúng |
| Resumed attempt | 12,203 giây ≈ 203.38 phút | Bao gồm các stage của lần resume, không riêng model.generate |
| Training | ~3,260 giây, 1,188 steps, 2 epochs | Ghi nhận từ run trước; chưa đọc lại original training log trong lượt lập plan |
| Tổng training + resumed attempt | Ít nhất ~257.7 phút | Chưa gồm probe và overhead của attempt thất bại |

Training candidate source là `cdf35a8...`; hotfix merge/reuse ở `683dab1`, source push tiếp ở `75c7a09`. Candidate/gates cũ không tự chứng minh inference chạy cùng code gốc. Cần lưu riêng training source và inference source trong manifest mới.

## 3. Những kết luận cũ bị rút lại

| Phát biểu cũ | Cách diễn giải đúng để lập kế hoạch |
|---|---|
| Top-1 thực tế 31.6%, v13 ~42% | Chưa có phép đo độc lập. Các số đó suy ra từ synthetic formula, không phải retrieval accuracy |
| Công thức khớp 100% chứng minh code đúng | Lập luận vòng tròn; không dùng để kiểm chứng code hay dự đoán leaderboard |
| Nguyên nhân 100% retrieval, tăng token cap tạo toàn bộ gain | Chưa có ablation; generator, prompt, checkpoint, memory và assembly đều là yếu tố gây nhiễu |
| encoder_ft_v2 MRR 0.3407 = Hit@1 34.1% | MRR và Hit@1 là hai thống kê khác nhau. Phải đọc metric key, tập và provenance |
| Thay encoder chắc tăng lên 55–65% / +0.066 | Giả thuyết chưa kiểm chứng; không đưa vào success criteria như kết quả mong đợi chắc chắn |
| Inference 32 phút, nhanh hơn 12 lần | Log resume ghi 203.38 phút toàn attempt; cần stage timing mới để đo riêng inference và speedup công bằng |
| train_loss 0.7797 là final batch loss; token accuracy chứng minh chất lượng cao | 0.7797 là aggregate train loss; teacher-forced accuracy không phải free-generation quality |
| Câu dài luôn gây thêm fragmentation penalty | Từ thừa có thể làm giảm precision; fragmentation phụ thuộc alignment cụ thể, không đơn điệu theo độ dài |
| 869 tokens ≈650 từ, gold luôn 349 từ | Không có tỷ lệ đổi token/từ cố định; phải tính lại trên cùng tập và tokenizer |
| Toàn bộ 19 lỗi đã fix, cache resume tức thì | Code hiện còn các lỗi bên dưới; chưa có interrupted-batch proof |
| >0.60 chắc chắn | Chỉ là mục tiêu; cần receipt mới để xác nhận |

## 4. Phát hiện trực tiếp từ code tại HEAD

Line numbers chỉ làm mốc của HEAD đã ghi; tìm bằng symbol khi implement.

| ID | Ưu tiên | Bằng chứng | Tác động / task |
|---|---|---|---|
| F01 | P0 | `scripts/run_gpu_gate.py:413–418` gán 0.482/0.518, `mini_eval_completed=True`; missing timing fallback 1.42 | Gate không chứng minh chất lượng; R1 |
| F02 | P0 | `src/task2/generation/trainer.py:411–430` collate raw dataset, catch rồi skip | Không xác minh masking thật; R2 |
| F03 | P0 | `trainer.py:369` warmup_steps=1; `GeneratorTrainConfig` chưa có warmup_ratio/scheduler fields | Recipe YAML khác thực thi; R2 |
| F04 | P0 | `src/task2/predict.py:704–724` generate tất cả rồi mới append, nuốt write error; nhánh đọc lỗi dùng `logger` chưa được định nghĩa trong module đã kiểm tra | Mất tiến độ khi ngắt, error path có thể lỗi tiếp; R3 |
| F05 | P0 | Hash cache = prompt + max_tokens + adapter_path | Đổi weights cùng path có thể hit sai; đổi path cùng weights lại miss; R3 |
| F06 | P0 | `runner.py:234–241`, `modal_app.py:595` reuse bằng file existence/latest directory | Cần xác minh checkpoint content/training recipe, không chỉ tên; R4 |
| F07 | P0 | `modal_app.py:719–728` catch upload/bundle error rồi result PASS | Tách compute/submission/release; R6 |
| F08 | P0 | `run_bundle.py:242–250` tạo log success, trainer_state tối giản, telemetry từ probe | Không thể trình bày như full-run raw log/state/telemetry; R6 |
| F09 | P0 | `generation/dataset.py:228` chỉ exclude một fold; `dataset/splits.py` có train/dev/lockbox | Nếu dùng dev/lockbox làm independent evaluation thì exclude cả hai khỏi supervised training; mã cũ Q1 |
| F10 | P1 | `evaluation.py:315–345` load generator mặc định, pipeline không truyền resolved retrieval recipe; token default384 | So sánh offline/production lệch; Q1 |
| F11 | P1 | `evaluation.py:36–43` thiếu dual_assembled trong generator-dependent families | Có thể từ chối generator dù family đó thắng; Q1 |
| F12 | P1 | `scripts/sweep_assembly.py` đọc prose/reference/evidence nhưng CLI quảng cáo gen_raw_cache | Raw cache không đủ để chấm; silent empty fields khiến kết quả vô nghĩa; Q4 |
| F13 | P1 | `src/common/dense.py:184` preprocessing phụ thuộc substring `dek21` trong model_name | Local `encoder_ft_v2` có thể bỏ tokenization; Q2 |
| F14 | P1 | `scripts/run_modal.sh` positional arg2; quoted `python3 -m modal`; tự dispatch T4 trên Modal | Sai CLI/mất kiểm soát chi phí; R5 |
| F15 | P1 | `modal_app.py:778` bắt parent dù stage root không có parent | Root launcher khác unit request builder; R5 |
| F16 | P1 | `src/common/rrf.py` cộng theo chunk_id; chưa article aggregation trong hàm này | Đề xuất experiment, không mặc định bug hay gain; Q3 |
| F17 | P1 | `docs/HF_MODEL_CARD.md` ghi current dense ft, v13 zero-shot, current score target; total cộng base nhưng thêm dòng adapter | Metadata không khớp run; R6 |

| F18 | P0 | `scripts/run_gpu_gate.py:423–424` đọc `cuda_0.peak_allocated_mb`, nhưng `snapshot_cuda_memory` trả `devices[0].max_allocated_mb` và chưa có max-reserved | Có thể ghi memory=0 dù GPU đã sử dụng; R1 sửa schema và đo thật |
| F19 | P0 | `src/common/dense.py:176` trả random normalized vectors khi SentenceTransformer thiếu, cùng nhánh với explicit mock | Production phải fail khi dependency/model thiếu, chỉ test mode được mock; Q2 |

P0 là mức ưu tiên lịch sử về độ tin cậy, không có nghĩa từng lỗi chắc làm giảm METEOR hoặc mọi experiment phải sửa hết trước. Chỉ sửa lỗi ảnh hưởng đường đang thử theo03. Bản vá merge-adapter đã tồn tại; kiểm tra real-path nhỏ nếu dùng, không sửa lại tùy tiện.

## 5. Còn chưa biết

- Exact artifact và training exposure của baseline 0.5486, gồm pre-trained adapter và encoder_ft_v2.
- Hiệu quả retrieval/answer trên sample mới với đúng serving hiện tại. Lịch sử HF đã có paired retrieval A/B trong07; chưa chạy lại bằng code hiện tại.
- Tác động riêng của epoch, token cap, prompt, QA memory và appended citations.
- Official scorer environment/tokenizer revision đầy đủ; test parity hiện chạy trong một process có thể dùng cùng upstream import cho cả hai phía.
- Quy tắc BTC về external models, parameter counting và deterministic post-processing. Không khẳng định hợp lệ nếu chưa đọc quy tắc áp dụng.

Các unknown không phải lý do bịa số, cũng không chặn toàn bộ vòng thử nhanh. Dùng checkpoint rõ danh tính, ghi diagnostic nếu exposure chưa biết; chỉ giải quyết unknown liên quan trực tiếp đến phép thử hoặc quy định của lần nộp bài.
