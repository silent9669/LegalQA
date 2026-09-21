import json
import pytest
from scripts.sweep_assembly import evaluate_assembly_strategies


def test_evaluate_assembly_strategies_ranks_candidates():
    # 2 synthetic items: ref has statutory text, poor prose has only a short stub
    items = [
        {
            "qa_id": "q1",
            "reference": "Căn cứ Điều 5 Luật An toàn thực phẩm 2010. Nghiêm cấm sản xuất thực phẩm giả.",
            "prose": "Nghiêm cấm sản xuất thực phẩm giả.",
            "evidence": "Điều 5. Các hành vi bị cấm\n1. Sản xuất thực phẩm giả.\n2. Kinh doanh thực phẩm không rõ nguồn gốc.",
            "doc_name": "Luật An toàn thực phẩm 2010",
            "art_num": "5",
        }
    ]
    results = evaluate_assembly_strategies(items, citation_budgets=[2000, 4000])
    assert "dual_assembled_4000" in results
    assert "generated" in results
    assert results["dual_assembled_4000"]["meteor"] >= results["generated"]["meteor"]
    assert results["dual_assembled_4000"]["mean_words"] > results["generated"]["mean_words"]
