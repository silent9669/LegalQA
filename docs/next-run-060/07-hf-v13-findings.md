# HF và v13 — model nào đã train, điểm nào đã đo?

**Ngày kiểm tra:** 2026-09-22. **Phạm vi:** lịch sử HF public, metadata/config, các cell/log liên quan của notebook thực thi và notebook đã xóa trong Git. Không tải weight binaries, không chạy lại training/inference, không audit toàn bộ từng dòng log. Không thể nhận diện người upload thật từ tài khoản dùng chung.

## 1. Nguồn có thể đối chiếu

Repository: [dangphuc2109/legalqa-qwen2.5-3b-adapter](https://huggingface.co/dangphuc2109/legalqa-qwen2.5-3b-adapter).

| Nguồn | Revision / vị trí | Ý nghĩa |
|---|---|---|
| Snapshot HF đọc lần này | `a6e4805b5a25dec3f3220f6a0a6506ecc5e00e53` | Pin nội dung thay vì đọc main thay đổi |
| Commit chứa private run | `6a2721e34a083eae202dbb80e4fe529707ec4097` | Message ghi `Private run 20260920-215402: encoder_ft_v2 + submission 1918` |
| [Executed notebook](https://huggingface.co/dangphuc2109/legalqa-qwen2.5-3b-adapter/blob/a6e4805b5a25dec3f3220f6a0a6506ecc5e00e53/runs/20260920-215402/executed.ipynb) | `runs/20260920-215402/executed.ipynb` | 26cells, title v13.2, có source và outputs |
| [FT metadata](https://huggingface.co/dangphuc2109/legalqa-qwen2.5-3b-adapter/blob/a6e4805b5a25dec3f3220f6a0a6506ecc5e00e53/runs/20260920-215402/encoder_ft_v2/ft_meta.json) | Cùng snapshot | Init/loss history/hyperparameters thực tế |
| Run config / logs | `runs/20260920-215402/run_config_v6.json`, `logs/launcher.out.log`, `logs/modal-app.log` | Assembly metrics, execution context, timings |
| Snapshot trước purge | `295d6ff206ce14a69412aa1cb1728849ffe56fde` | Adapter/manifest cũ vẫn truy xuất được dù main đã xóa |
| Notebook trong Git | `5f75cfa:notebooks/DSC2026_LegalQA_Pipeline_v13.ipynb` | Source26cells, **không có execution outputs**; bị xóa ở `cdf35a8` |

Hashes và pinned URLs: [hf-inspection-sources.json](evidence/hf-inspection-sources.json). Bản sao metadata nhỏ: [encoder config](evidence/encoder-ft-v2-config.json), [ft_meta](evidence/encoder-ft-v2-meta.json). Không copy raw notebook/log lớn vào repo để tránh chứa dữ liệu/credential ngoài scope.

Các cell dưới đây đánh số **zero-based**. Source comment, config dự định, execution output, local metric và official receipt là năm loại bằng chứng khác nhau.

## 2. “forbert” là PhoBERT-family dense retriever

Encoder ID: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`. Export config ghi:

- `architectures=[RobertaModel]`, `tokenizer_class=PhobertTokenizer`.
- Hidden768,12layers,12heads,vocab64001,max_position_embeddings258.
- Notebook cap256 để dành RoBERTa position offsets.
- Sentence representation: attention-masked **mean pooling + L2 normalize**; PyVi word segmentation tương thích train/serving.

Không phải model sinh đáp án. Generator vẫn là Qwen2.5-3B-Instruct + PEFT adapter. Cùng dimension768 không đủ để kết luận hai index tương thích.

## 3. Dense training thực thi — cell11 / CELL6d

| Thuộc tính | Log/metadata |
|---|---|
| Init thực tế | Base DEk21, LR2e-5 |
| Objective | MatryoshkaLoss(MultipleNegativesRankingLoss), dims768/512/256 |
| Training rows | 4,166; mỗi row một question, một positive, hai negatives |
| Epochs/batch/maxlen | 2 /64 /256 |
| Warmup/sampler | Ratio0.10; NO_DUPLICATES, drop_last |
| Held-out exclusion | 699validation IDs và698normalized question texts |
| Mining | Hybrid top100; reranker chấm tối đa12negative candidates |
| Denoising | Exclude gold docs, bỏ near-copy Jaccard>0.60, veto negative score≥positive score |
| Negative accounting | 24,333probable false negatives bị veto;5,291hard +3,041random fallback =8,332negatives giữ lại |

**Không phải bằng chứng continuation:** source có `FT_TAG='ft_v2'`, `FT_INIT='current'`, LR continuation1e-5; nhưng branch fallback chọn base. Output ghi “initialising from the base encoder · lr 2e-05”, và `ft_meta.init` cũng là base. Vì vậy “train nối từ ft_v2 thêm1epoch” trong04 là **thử nghiệm đề xuất mới**, không phải lặp lại kết quả đã chứng minh.

Giới hạn: một positive chọn theo answer overlap là heuristic; NO_DUPLICATES không loại hết semantic false negatives. Random fallback không trải qua toàn bộ reranker/Jaccard denoising như hard negatives.

## 4. Thời gian — đọc đúng phần nào nhanh

Hardware thực thi: **NVIDIA A100 80GB PCIe**; Torch2.5.1+cu124, Transformers4.49.0, SentenceTransformers3.4.1, BF16/SDPA.

| Phần việc | Thời gian ghi trong notebook |
|---|---|
| Chuẩn bị/mine training rows | 420s, khoảng7phút |
| Gradient training encoder | 1.2phút |
| Corpus encoding | 4.9phút |
| Toàn bộ retriever cell | 17.3phút |
| Generation validation120câu ở cell17 | 10.5phút |

17.3phút bao gồm thêm A/B/export/checks ngoài ba mục đầu; không cộng riêng các timer rồi gọi đó là total. Corpus801,863physical passages,788,498unique texts,20shards. Lịch sử dùng23PyVi workers; không copy worker count này sang Kaggle ít CPU.

**Kết luận về tốc độ:** bỏ Qwen retraining có thể tránh phần việc lớn, nhưng toàn vòng thử không chỉ1.2phút. Chưa đo ETA tương ứng trên T4/A10040GB. Notebook source trong Git có comment “~25min”; đó không phải số đo mới.

## 5. Retrieval A/B đã ghi nhận

Cell11 so encoder base với trained encoder:

| Metric | Base | Fine-tuned | Sample |
|---|---:|---:|---:|
| Dense Hit@1 | 0.2175 | 0.2675 | 400 |
| Dense Hit@8 | 0.4500 | 0.4700 | 400 |
| Dense MRR | 0.29003869 | 0.34075 | 400 |
| Hybrid Hit@1 | 0.2275 | 0.2650 | 400 |
| Hybrid Hit@8 | 0.4350 | 0.4725 | 400 |
| Hybrid MRR | 0.29752083 | 0.33139583 | 400 |
| Reranked Hit@1 | 0.34666667 | 0.36666667 | 150 |
| Reranked Hit@8 | 0.56666667 | 0.5800 | 150 |
| Reranked MRR | 0.41364286 | 0.43564286 | 150 |

Notebook adopted ft_v2. **0.34075 là MRR, không phải Hit@1.** Không suy ra METEOR tăng một lượng cố định từ các chỉ số này. Retrieval lab cell12 có pipeline/sample khác; không trộn số của nó vào bảng paired này.

Round-trip checks ghi cosine minimum0.99997 giữa train/export reload; stored-corpus vs serving trên48passages minimum0.9973. Đây là checks **trong lịch sử**, không phải checks đã chạy lại hôm nay.

## 6. Qwen đã được reuse, không SFT tại đây

Cell15 / CELL T4 ghi generator training T1/T2/T3 không có trong notebook; loader thực tế:

```python
USE_TRAINED_ADAPTER = False
ADAPTER_IN_USE = ADAPTER_DIR
model = PeftModel.from_pretrained(model, ADAPTER_IN_USE, torch_dtype=DTYPE)
model = model.merge_and_unload()
```

Output: `PRE-TRAINED (CELL 0) → /vol/kaggle_data/lora`. **False nghĩa dùng adapter có sẵn, không phải bỏ adapter.** BF16, SDPA, greedy; merged generator khoảng3.086B theo log, chưa phải phép đếm compliance toàn hệ thống.

### Adapter cũ nào? Chưa đủ bằng chứng để gọi tên best checkpoint

Ở snapshot pre-purge, `checkpoints/generator/hf_adapter/adapter_model.safetensors` có119,801,528bytes; LFS SHA256 `1a0a16f5033b9d2740f790fcc09e8c5b997f279397cc164fcb842e0119158c4a`. Cùng blob ở `runs/checkpoint-3/` bên trong adapter directory.

Nhưng `generator_manifest.json` ghi dataset10, optimizer3steps,max_seq_len256; `trainer_state.json` ghi global_step=max_steps=3. Đây là metadata giống smoke/probe, **không phải bằng chứng adapter được train full**. Chưa biết liệu từng kế thừa weights trước đó hay không.

Notebook execution signature ghi `119801528:f89116e19fa4`. Fingerprint này là SHA1 rút gọn của phần đầu/cuối file, không phải full SHA256. Cùng kích thước không chứng minh cùng weights; chưa đối chiếu fingerprint bytes. Không khẳng định adapter pre-purge chính là `/vol/kaggle_data/lora` của run có outputs. Dùng checkpoint production rõ lineage cho baseline mới, thay vì đoán từ folder name.

## 7. Assembly — có cơ hội rẻ hơn retraining

Cell17/19/20, cùng120validation questions trong notebook; local scores, không phải official:

| Variant | Local METEOR |
|---|---:|
| Whole article4000chars | 0.521703635 |
| Whole article6000chars | 0.524496542 |
| Whole article9000chars | khoảng0.5224 |
| Supervised clause selection cap800 | 0.522640325 |
| Citation-only cap260 | 0.376951444 |
| Dedup threshold0.95 | 0.512170354 |
| Dedup threshold0.80 | 0.502847848 |
| Answer cap250 | 0.425303368 |
| Answer cap450 | 0.481427034 |

`vi_tokens(s)` trong cell16 là NFC normalization + lowercase + `.split()`: các cap kiểu này là **whitespace metric tokens**, không phải Qwen model tokens. Citation budget4000/6000 lại tính **characters**. Vì vậy không đổi “869tokens” thành “khoảng650words” như báo cáo cũ.

Ridge selector cell20 fit3,000articles →47,542clauses; excludes699validation questions. Delta supervised800 so4000 chỉ khoảng+0.000937 ở sample này. Không phải bước nhảy lớn đã được chứng minh. Final notebook chọn supervised800, có82exact known-QA overrides và0fuzzy; không tự coi giảm fuzzy threshold là cải thiện.

Thử lại4000vs6000 và complete-clause policy bằng cached prose là đáng làm vì rẻ, **không vì đã chắc sẽ thắng trên tập khác**. Các “projected Codabench” trong notebook chỉ là heuristic, không dùng làm forecast.

## 8. “Mỗi lần train điểm tăng” — phần nào được xác nhận?

- Đã xác nhận **một paired dense A/B cải thiện retrieval** và nhiều phép assembly comparison trên output có sẵn.
- Các pre-purge manifests ghi local METEOR:20260915≈0.496963;20260916≈0.484123;20260917≈0.484123. Không tạo thành đường tăng đơn điệu. Đây cũng không phải chuỗi official submissions cùng protocol.
- Notebook Git source có comments gắn0.5224/0.5486 với assembly variants và kể một run0.5495 không generate mới do cache cũ. **Đó là comment lịch sử, chưa được receipt/log tương ứng xác nhận**; chỉ nhắc rằng điểm tăng có thể do assembly/cache chứ không phải training mới.
- `runs/run_v16_dual_assembled/submission_manifest.json` ghi target>0.60, encoderft_v2 và adapter từrun0.49. Target không phải achieved score; manifest không chứng minh serving đã đúng.
- Benchmark0.5486 hiện là user/model-card context; receipt mới trực tiếp xác nhận trong folder này vẫn là official0.524640878.
- Tất cả HF author fields được thấy đều thuộc cùng tài khoản. Theo người dùng, teammate dùng API credential của tài khoản đó. **Chưa xác định chắc “hai commit” cụ thể hoặc ai thực sự chạy/upload từng run.** Không cần chờ giải quyết attribution để dùng recipe đã kiểm tra.

Trong output notebook, embedded HF push còn lỗi404sai destination; bundle hiện có trên repo không chứng minh upload cell đó đã thành công. Không copy upload/token cells vào experiment mới.

## 9. Quyết định sau nghiên cứu

**Nên lấy từ v13:** retriever nhỏ + denoised negatives, explicit mean/L2/segmentation, reuse Qwen, cache output để thử assembly, sample validation cố định và index parity nhỏ.

**Không lấy nguyên:** dynamic adapter auto-pick, cache chỉ dựa QA ID, dự báo leaderboard bằng offset/công thức, claim zero-shot sai, continuation suy từ tagv2, ép độ dài, chuỗi kiểm định dài.

Đường đề xuất: reuse ft_v2 → assembly sweep → một dense continuation nếu có lý do → chỉ xét Qwen training khi lỗi sinh thực tế đòi hỏi. Chi tiết [04](04-quality-implementation-plan.md), [05](05-experiment-matrix.md), [06](06-release-and-a100-runbook.md).

### Tài liệu ngoài đã tham khảo

- [DEk21 model card](https://huggingface.co/CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2): backbone bi-encoder tiếng Việt, PyVi, cosine, Matryoshka. Recipe cached MNRL trên card là upstream model training, khác ordinary MNRL của fine-tune notebook đã inspect.
- [Sentence Transformers losses guidance](https://github.com/huggingface/sentence-transformers/blob/main/skills/train-sentence-transformers/references/losses_sentence_transformer.md): cached MNRL/GradCache và contrastive batch semantics.
- [Sentence Transformers efficiency](https://github.com/huggingface/sentence-transformers/blob/main/docs/sentence_transformer/usage/efficiency.rst): precision/batching efficiency. Tài liệu main không đảm bảo API tương thích historical3.4.1; đây là hướng tham khảo, chưa là dependency change.
