from src.common.legal_parser import parse_legal_document

def test_parse_legal_document():
    passage = """Nghị định 90/2017/NĐ-CP
Chương I. Quy định chung
Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về xử phạt vi phạm hành chính.
Điều 17. Vi phạm về tiêm phòng
1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm phòng.
2. Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với hành vi che giấu dịch bệnh."""

    chunks = parse_legal_document(doc_id="740", doc_name="Nghị định 90/2017/NĐ-CP", passage=passage)
    assert len(chunks) >= 3
    art17_chunks = [c for c in chunks if c["article_number"] == "17"]
    assert len(art17_chunks) >= 2
    assert all(c["parent_article_id"] == "doc740_art17" for c in art17_chunks)
    assert any("không tiêm phòng" in c["text_raw"] for c in art17_chunks)
