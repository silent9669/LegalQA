import argparse
import json
import os
import sys
import zipfile
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.task2.predict import LegalQAPipeline

def save_submission_artifacts(results: dict, output_json: str, zip_path: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(output_json, arcname="submission.json")

def main():
    parser = argparse.ArgumentParser(description="LegalQA Task 2 Inference Script")
    parser.add_argument("--input", default="artifacts/raw/public-official.json", help="Input questions JSON path")
    parser.add_argument("--output", default="artifacts/task2/submissions/submission.json", help="Output submission JSON path")
    parser.add_argument("--max_tokens", type=int, default=180, help="Max new tokens per generation")
    args = parser.parse_args()

    print(f"Loading questions from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                items.append({"id": str(k), "question": v.get("question", "")})
            else:
                items.append({"id": str(k), "question": str(v)})
    elif isinstance(data, list):
        items = [{"id": str(r.get("id") or r.get("qa_id")), "question": str(r.get("question", ""))} for r in data]

    print(f"Initializing LegalQA pipeline with {len(items)} queries from full index...")
    pipeline = LegalQAPipeline.load_pipeline(
        data_dir="artifacts/task2/data",
        index_dir="artifacts/task2/indexes/bm25"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    jsonl_path = args.output + "l"
    zip_path = args.output + ".zip"
    results = {}

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if len(r.get("answer", "")) > 30:
                            results[str(r["id"])] = {"answer": r["answer"]}
                    except Exception:
                        pass
        print(f"Resuming: {len(results)} queries already loaded from {jsonl_path}")

    # Immediately populate all 1,000 queries with grounded answers
    temp_full_results = {}
    print("Generating baseline grounded answers for initial submission package...")
    for item in items:
        qid = str(item["id"])
        if qid in results:
            temp_full_results[qid] = results[qid]
        else:
            exact_mem = pipeline.memory.lookup_exact(qid, item["question"])
            if exact_mem:
                temp_full_results[qid] = {"answer": exact_mem}
                results[qid] = {"answer": exact_mem}
            else:
                # Fast grounded extractive answer with Strategy F
                seeds = pipeline.bm25.search(item["question"], top_k=3)
                stitched = pipeline.stitcher.stitch(seeds)
                doc_name = seeds[0].get("doc_name", "") if seeds else ""
                art_num = seeds[0].get("article_number", "") if seeds else ""
                clause_num = seeds[0].get("clause_number", "") if seeds else ""
                from src.task2.source_snap import build_citation_header, clean_statutory_text, apply_strategy_f
                header = build_citation_header(doc_name, art_num, clause_num)
                raw_chunk = clean_statutory_text(stitched.get("stitched_text", ""))
                base_ans = f"{header}\n{raw_chunk[:600]}"
                temp_full_results[qid] = {"answer": apply_strategy_f(base_ans, raw_chunk, max_chars=1500)}

    save_submission_artifacts(temp_full_results, args.output, zip_path)
    print(f"Saved initial valid submission.json and submission.json.zip with {len(temp_full_results)} entries.")

    with open(jsonl_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(tqdm(items, desc="Generating Neural Predictions"), start=1):
            qa_id = str(item["id"])
            if qa_id in results and len(results[qa_id]["answer"]) > 100:
                continue
            q = item["question"]
            ans = pipeline.predict_single(qa_id, q, max_new_tokens=args.max_tokens)
            results[qa_id] = {"answer": ans}
            temp_full_results[qa_id] = {"answer": ans}
            f_out.write(json.dumps({"id": qa_id, "answer": ans}, ensure_ascii=False) + "\n")
            f_out.flush()

            if idx % 10 == 0 or idx == len(items):
                save_submission_artifacts(temp_full_results, args.output, zip_path)

    save_submission_artifacts(results, args.output, zip_path)
    print(f"Final submission complete: {args.output} and {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
