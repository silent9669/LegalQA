from src.task2.evidence_packer import EvidencePacker


def test_evidence_packer_variants():
    chunks = [
        {
            "chunk_id": "c1_p1",
            "doc_name": "Nghị định 90/2017/NĐ-CP",
            "parent_article_id": "doc1_art17",
            "article_number": "17",
            "clause_number": "1",
            "start_char": 0,
            "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Khoản 1: Phạt cảnh cáo đối với hành vi A.",
        },
        {
            "chunk_id": "c1_p2",
            "doc_name": "Nghị định 90/2017/NĐ-CP",
            "parent_article_id": "doc1_art17",
            "article_number": "17",
            "clause_number": "2",
            "start_char": 100,
            "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Khoản 2: Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi B.",
        },
        {
            "chunk_id": "c2_p1",
            "doc_name": "Nghị định 100/2019/NĐ-CP",
            "parent_article_id": "doc2_art5",
            "article_number": "5",
            "clause_number": "1",
            "start_char": 0,
            "text_raw": "[DOCUMENT] Nghị định 100/2019/NĐ-CP\n[ARTICLE] Điều 5. Khoản 1: Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng.",
        },
    ]

    packer = EvidencePacker(chunks)

    # 1. Focused Clause
    pack_focused = packer.pack_evidence(chunks[:2], pack_type="focused_clause")
    assert pack_focused["pack_type"] == "focused_clause"
    assert "Khoản 1" in pack_focused["text"]
    assert "Khoản 2" not in pack_focused["text"]

    # 2. Primary Article
    pack_art = packer.pack_evidence(chunks[:2], pack_type="primary_full_article")
    assert "Khoản 1" in pack_art["text"]
    assert "Khoản 2" in pack_art["text"]
    assert len(pack_art["clause_ids"]) == 2

    # 3. Multi-Seed Pack (no duplicate clauses)
    pack_multi = packer.pack_evidence(chunks, pack_type="multi_seed_2500_chars")
    assert "Nghị định 90/2017" in pack_multi["text"]
    assert "Nghị định 100/2019" in pack_multi["text"]
    assert pack_multi["chars"] <= 2500


def test_evidence_packer_no_duplicate_clauses():
    # Provide identical seed multiple times in retrieval results
    chunk = {
        "chunk_id": "c1",
        "doc_name": "Luật Doanh nghiệp",
        "parent_article_id": "art_1",
        "article_number": "1",
        "clause_number": "1",
        "text_raw": "Điều 1. Phạm vi điều chỉnh",
    }
    packer = EvidencePacker([chunk])
    seeds = [chunk, chunk, chunk]

    pack = packer.pack_evidence(seeds, pack_type="multi_seed_2500_chars")
    assert pack["text"].count("Điều 1. Phạm vi điều chỉnh") == 1
