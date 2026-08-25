class CandidateSelector:
    """
    Candidate Selector that picks the highest expected METEOR score candidate
    among candidate_generate, candidate_snap, candidate_extract, and candidate_memory.
    """
    def __init__(self, mode: str = "rules"):
        self.mode = mode

    def select(self, question: str, candidates: dict[str, str], features: dict = None) -> str:
        features = features or {}

        if "candidate_memory" in candidates and candidates["candidate_memory"]:
            return candidates["candidate_memory"]

        # If strong extractive match for penalty / numeric questions
        if features.get("has_penalty_keyword") and features.get("extractive_quality", 0) > 0.85:
            if "candidate_extract" in candidates and candidates["candidate_extract"]:
                return candidates["candidate_extract"]

        # Default preference: candidate_snap > candidate_generate > candidate_extract
        for key in ["candidate_snap", "candidate_generate", "candidate_extract"]:
            if key in candidates and candidates[key]:
                return candidates[key]

        return list(candidates.values())[0] if candidates else ""
