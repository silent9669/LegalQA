def build_generation_prompt(question: str, evidence_chunks: list[dict], examples: list[dict] = None) -> str:
    """
    Constructs an authoritative, citation-focused prompt for the LLM generator.
    """
    prompt_lines = [
        "Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các căn cứ pháp lý được cung cấp.",
        "Yêu cầu: Trả lời chính xác, trích dẫn rõ căn cứ (Điều, khoản, tên văn bản) và giữ đúng các số liệu, thời hạn, mức phạt.",
        "",
        "### Căn cứ pháp lý:"
    ]
    for idx, chunk in enumerate(evidence_chunks[:6], 1):
        prompt_lines.append(f"[{idx}] {chunk.get('raw_text', '')}")
        prompt_lines.append("")

    if examples:
        prompt_lines.append("### Ví dụ tham khảo:")
        for ex in examples[:2]:
            prompt_lines.append(f"Câu hỏi: {ex.get('question')}")
            prompt_lines.append(f"Trả lời: {ex.get('answer')}")
            prompt_lines.append("")

    prompt_lines.append("### Câu hỏi:")
    prompt_lines.append(question)
    prompt_lines.append("")
    prompt_lines.append("### Trả lời:")
    return "\n".join(prompt_lines)
