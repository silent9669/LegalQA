import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.provision_memory import ProvisionMemory

def test_provision_memory():
    prov_data = {
        "13/2023/NĐ-CP::43": [
            {"id": "132819", "question": "NĐ 13 áp dụng từ ngày nào?", "answer": "Căn cứ Điều 43..."}
        ]
    }
    pm = ProvisionMemory(prov_data)
    exs = pm.lookup("13/2023/NĐ-CP", "43")
    assert len(exs) == 1
    assert exs[0]["id"] == "132819"
    assert len(pm.lookup("13/2023/NĐ-CP", "99")) == 0
