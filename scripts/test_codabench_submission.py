import os
import sys
import json

def test_codabench_simulation(submission_path: str = "artifacts/submissions/submission.json"):
    print("=== Testing CodaBench Compatibility Simulation ===")

    if not os.path.exists(submission_path):
        if os.path.exists("submission.json"):
            submission_path = "submission.json"
        else:
            print(f"Error: {submission_path} does not exist.")
            return False

    with open(submission_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Submission path: {submission_path}")
    print(f"Submission total keys: {len(data)}")
    sample_key = list(data.keys())[0]
    sample_val = data[sample_key]
    print(f"Sample item {sample_key}:")
    print(f"  {sample_val['answer'][:200]}...")

    # Validate structure: {"<id>": {"answer": "<str>"}}
    valid_format = True
    for k, v in list(data.items()):
        if not isinstance(v, dict) or "answer" not in v:
            valid_format = False
            print(f"Format error on key {k}: {v}")
            break
        if not isinstance(v["answer"], str) or not v["answer"].strip():
            valid_format = False
            print(f"Answer type/empty error on key {k}: {type(v['answer'])}")
            break

    if valid_format:
        print("\n★ PASS: Submission JSON format strictly matches CodaBench schema!")
    else:
        print("\n❌ FAIL: Submission JSON failed schema check.")
        return False

    return True

if __name__ == "__main__":
    test_codabench_simulation()
