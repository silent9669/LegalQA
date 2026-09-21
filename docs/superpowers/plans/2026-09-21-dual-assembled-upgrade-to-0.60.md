# LegalQA Dual-Part Assembly Engine & Upgraded Pipeline Plan (Target > 0.60)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the dual-part statutory answer assembly engine (`prose + "\n\nTrích dẫn quy định:\n" + citation_block`), set repetition penalty to 1.0, structure the Hugging Face repository run history, and validate the pipeline to elevate performance from 0.49/0.55 toward 0.60+.

**Architecture:** 
The official competition evaluation metric is METEOR with $\alpha = 0.9$ (90% weight on Recall). Reference answers contain both analytical prose and full statutory citations.
We integrate:
1. `clean_model_prose`: Strips LLM preambles, fences, and collapses loops.
2. `dual_assembled` candidate: Pairs the model's generated legal reasoning with the complete primary statutory article (`Trích dẫn quy định:\n`), reaching the optimal ~850-word length.
3. Repetition penalty = 1.0 in `QwenGenerator` to avoid penalizing verbatim statute quotes.
4. Clean Hugging Face repository layout documenting baseline (0.5486), QLoRA foundation (loss 0.7726), and the upgraded release.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, Transformers 5.0.0, PEFT 0.19.1, Hugging Face Hub, Modal A100.

**Spec:** `docs/ARCHITECTURE.md`, `configs/task2/algorithm.yaml`, `configs/task2/runtime/modal_a100.yaml`.

## Global Constraints

- Never mutate protected algorithm fields (`max_seq_len: 2048`, `quantization: 4bit_nf4`, `effective_batch_size: 8`).
- Effective batch size invariant: `per_device_train_batch_size * gradient_accumulation_steps == 8`.
- Total learned parameters must remain < 4.0B (current: 3.788B, margin: 211M).
- Pre-push check (`./test.sh`) must remain 100% PASS with 0 parameter violations and 0 secrets leaked.

---

### Task 1: Implement Dual-Part Statutory Answer Assembly in Candidates and Selector

**Files:**
- Modify: `src/task2/candidates.py`
- Modify: `src/task2/production_config.py`
- Modify: `src/task2/selector.py`
- Test: `tests/unit/test_dual_assembled_candidate.py`

**Interfaces:**
- Consumes: `snap_facts_to_evidence`, `clean_statutory_text`, `evidence_packs`
- Produces: `candidates["dual_assembled"]` containing `f"{prose}\n\nTrích dẫn quy định:\n{citation_block}"`

- [ ] **Step 1: Write failing test for `dual_assembled` candidate generation**

```python
def test_dual_assembled_candidate_structure():
    from src.task2.candidates import generate_candidate_ensemble

    gen_ans = "Theo quy định tại Điều 17 Nghị định 90/2017/NĐ-CP, người vận chuyển động vật bị phạt từ 6 đến 8 triệu đồng."
    ev_packs = {
        "full_article": "Điều 17. Vi phạm quy định về kiểm dịch động vật\n1. Phạt tiền từ 4.000.000 đồng...\n3. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng..."
    }
    cands = generate_candidate_ensemble(
        gen_ans=gen_ans,
        evidence=ev_packs["full_article"],
        doc_name="Nghị định 90/2017/NĐ-CP",
        art_num="17",
        evidence_packs=ev_packs,
    )

    assert "dual_assembled" in cands
    ans = cands["dual_assembled"]
    assert "Trích dẫn quy định:" in ans
    assert "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP quy định như sau:" in ans
    assert "Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng" in ans
    assert ans.startswith(gen_ans)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv311/bin/pytest tests/unit/test_dual_assembled_candidate.py -v`
Expected: FAIL with `AssertionError: assert 'dual_assembled' in cands`

- [ ] **Step 3: Implement `clean_model_prose` and `dual_assembled` in `src/task2/candidates.py`**

Incorporate preamble stripping regex (`_PREAMBLE_REGEX`), clean prose, and append statutory citation block into `candidates["dual_assembled"]`. Register `"dual_assembled"` in `GENERATOR_DEPENDENT_CANDIDATES` (`src/task2/production_config.py`) and `CANDIDATE_ORDER` (`src/task2/selector.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv311/bin/pytest tests/unit/test_dual_assembled_candidate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/task2/candidates.py src/task2/production_config.py src/task2/selector.py tests/unit/test_dual_assembled_candidate.py
git commit -m "feat(candidates): implement dual-part answer assembly (prose + statutory citation)"
```

---

### Task 2: Repetition Penalty Calibration and Batch Pacing in `QwenGenerator`

**Files:**
- Modify: `src/task2/generator.py:315-325`
- Test: `tests/unit/test_batch_oom_backoff.py`

**Interfaces:**
- Consumes: `repetition_penalty = 1.0`
- Produces: Verbatim evidence quotes are unpenalized during beam/greedy generation

- [ ] **Step 1: Write test verifying repetition penalty 1.0 in generator options**

- [ ] **Step 2: Update `src/task2/generator.py` to use `repetition_penalty=1.0`**

Set `repetition_penalty = 1.0` in `_generate_texts` so model can quote statutes verbatim from the prompt.

- [ ] **Step 3: Run generator unit tests**

Run: `./.venv311/bin/pytest tests/unit/test_batch_oom_backoff.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/task2/generator.py
git commit -m "perf(generator): set repetition_penalty to 1.0 for unpenalized statute quotation"
```

---

### Task 3: Configure Modal App Production Selection to `dual_assembled`

**Files:**
- Modify: `scripts/modal_app.py:215-235`
- Modify: `tests/launchers/test_modal_app.py`

**Interfaces:**
- Consumes: `build_remote_production_cfg`
- Produces: `best_fixed_candidate="dual_assembled"` with `max_new_tokens=512`

- [ ] **Step 1: Update `build_remote_production_cfg` in `scripts/modal_app.py`**

Set `best_fixed_candidate="dual_assembled"` and update `test_modal_app.py` asserts.

- [ ] **Step 2: Run launcher tests**

Run: `./.venv311/bin/pytest tests/launchers/test_modal_app.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/modal_app.py tests/launchers/test_modal_app.py
git commit -m "feat(modal): configure production profile to deploy dual_assembled candidate"
```

---

### Task 4: Full Pre-Push Test Gate Validation

**Files:**
- Tool: `./test.sh`

- [ ] **Step 1: Run `./test.sh`**

Run: `./test.sh`
Expected: ALL unit, integration, contract, and dataset tests PASS in < 50s.
Parameter compliance verified (< 4.0B).
Working tree clean.
