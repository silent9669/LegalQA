import pytest
from src.task2.generation.dataset import (
    SFTExample,
    truncate_evidence_preserving_answer,
    build_sft_example_token_aware,
    build_grounded_training_examples,
    select_worst_case_probe,
)


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<|endoftext|>"

    def encode(self, text, add_special_tokens=False):
        # 1 character approx 0.5 words or split on whitespace
        return [1] * max(1, len(text.split()))

    def decode(self, token_ids, skip_special_tokens=True):
        return "word " * len(token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def test_truncate_evidence_preserving_answer_leaves_gold_answer_intact():
    q = "What is Article 5?"
    ans = "Article 5 defines the primary statutory obligation."
    ev = "\n\n".join([f"Paragraph {i}: Long legal discussion about obligations." for i in range(50)])

    res = truncate_evidence_preserving_answer(q, ev, ans, max_chars=200)
    assert len(res) <= 200
    assert "Paragraph 0:" in res


def test_build_sft_example_token_aware_preserves_gold_answer():
    tok = MockTokenizer()
    q = "Can a contract be rescinded?"
    ans = "Yes, under specific conditions set forth in Civil Code Article 420."
    ev = " ".join(["evidence"] * 500)

    full_text, diag = build_sft_example_token_aware(
        question=q,
        evidence_text=ev,
        answer=ans,
        tokenizer=tok,
        max_seq_len=2048,
    )

    assert not diag["dropped"]
    assert not diag["answer_truncated"]
    assert ans in full_text
    assert diag["total_tokens"] <= 2048


def test_build_sft_example_token_aware_drops_only_oversized_answer():
    tok = MockTokenizer()
    q = "Question"
    ans = " ".join(["long_answer"] * 3000)
    ev = "evidence"

    full_text, diag = build_sft_example_token_aware(
        question=q,
        evidence_text=ev,
        answer=ans,
        tokenizer=tok,
        max_seq_len=2048,
    )

    assert diag["dropped"] is True
    assert full_text is None


def test_select_worst_case_probe_deterministic_selection():
    examples = []
    for i in range(40):
        examples.append(
            SFTExample(
                prompt=f"prompt {i}",
                completion=f"completion {i}",
                total_tokens=100 + i * 10,
                completion_tokens=50 + (i % 5) * 20,
                qa_id=f"qa_{i}",
                text=f"text {i}",
            )
        )

    probe = select_worst_case_probe(examples, n_total=12, n_completion=12)
    assert len(probe) >= 24
    # Ensure highest total_tokens (e.g. qa_39, qa_38, etc.) are in probe
    probe_ids = {ex.qa_id for ex in probe}
    for i in range(28, 40):
        assert f"qa_{i}" in probe_ids

    # Calling again yields identical deterministic result
    probe_2 = select_worst_case_probe(examples, n_total=12, n_completion=12)
    assert [ex.qa_id for ex in probe] == [ex.qa_id for ex in probe_2]


def test_select_worst_case_probe_dict_compatibility():
    examples = [
        {"qa_id": "1", "total_tokens": 500, "completion_tokens": 100},
        {"qa_id": "2", "total_tokens": 1500, "completion_tokens": 200},
        {"qa_id": "3", "total_tokens": 800, "completion_tokens": 400},
    ]
    probe = select_worst_case_probe(examples, n_total=1, n_completion=1)
    assert len(probe) == 2
    qa_ids = {ex["qa_id"] for ex in probe}
    assert "2" in qa_ids  # top total
    assert "3" in qa_ids  # top completion
