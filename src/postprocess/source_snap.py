import re

AMOUNT_PATTERN = re.compile(r'([0-9]{1,3}(?:\.[0-9]{3})+(?:\s*(?:đồng|triệu đồng|nghìn đồng)))')

def source_snap_answer(generated_text: str, evidence_chunks: list[dict]) -> str:
    """
    Aligns and replaces paraphrased/informal numbers and monetary amounts in generated text
    with the exact verbatim spans found in the legal evidence to maximize METEOR score.
    """
    if not evidence_chunks or not generated_text:
        return generated_text

    all_evidence_text = " ".join([c.get("content", "") for c in evidence_chunks])
    evidence_amounts = AMOUNT_PATTERN.findall(all_evidence_text)

    snapped_text = generated_text
    for amt in evidence_amounts:
        num_raw = amt.split()[0].replace('.', '')
        if num_raw.endswith('000000'):
            millions = num_raw[:-6]
            colloquial_pattern = re.compile(rf'\b{millions}\s*triệu(?:\s*đồng)?\b', re.IGNORECASE)
            snapped_text = colloquial_pattern.sub(amt, snapped_text)

    return snapped_text
