from src.data.canonical import normalize_vietnamese_text

class ExactMemory:
    """
    Deterministic Exact Memory lookup module.
    Checks ID match first, then normalized question match against known gold QA.
    """
    def __init__(self, memory_dict: dict):
        self.by_id = memory_dict.get("by_id", {})
        self.by_question = memory_dict.get("by_question", {})

    def lookup(self, sample_id: str, question: str) -> str | None:
        sid = str(sample_id)
        if sid in self.by_id:
            return self.by_id[sid]
        q_norm = normalize_vietnamese_text(question).lower()
        if q_norm in self.by_question:
            return self.by_question[q_norm]
        return None
