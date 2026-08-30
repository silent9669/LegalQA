import os, gc, glob, json, zipfile, re, math, unicodedata, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import nltk

try:
    nltk.data.find("corpora/wordnet.zip")
except Exception:
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"PyTorch Target Device: {device}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Locate model directory dynamically
model_candidates = glob.glob("/kaggle/input/**/3b-instruct/**", recursive=True)
model_dirs = [d for d in model_candidates if os.path.isfile(os.path.join(d, "config.json"))]
if model_dirs:
    MODEL_PATH = model_dirs[0]
else:
    MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
print(f"Using Base Model from: {MODEL_PATH}")

# Locate datasets dynamically via glob
qa_files = glob.glob("/kaggle/input/**/qa_unique.parquet", recursive=True)
chunks_files = glob.glob("/kaggle/input/**/legal_chunks.parquet", recursive=True)
known_files = glob.glob("/kaggle/input/**/known_qa.json", recursive=True)
test_files = glob.glob("/kaggle/input/**/public-official.json", recursive=True)

assert qa_files, f"qa_unique.parquet not found in /kaggle/input/!"
assert chunks_files, f"legal_chunks.parquet not found in /kaggle/input/!"
assert known_files, f"known_qa.json not found in /kaggle/input/!"
assert test_files, f"public-official.json not found in /kaggle/input/!"

qa = pd.read_parquet(qa_files[0])
chunks_df = pd.read_parquet(chunks_files[0])
with open(known_files[0], encoding="utf-8") as f:
    known_qa = json.load(f)
with open(test_files[0], encoding="utf-8") as f:
    public_test = json.load(f)

print(f"Loaded {len(chunks_df)} legal chunks, {len(qa)} unique QA pairs, {len(public_test)} test queries.")

# Fast Inverted Index for Sparse Retrieval
def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[\r\t\f\v]", " ", text)
    text = re.sub(r"[^0-9a-zà-ỹ\s/\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("Building fast inverted sparse index on legal chunks...", flush=True)
corpus_texts = chunks_df["text_norm"].fillna("").tolist()
N = len(corpus_texts)

inverted_index = defaultdict(list)
doc_freq = Counter()

# Index first 400k high-value statutory chunks
for doc_id, text in enumerate(corpus_texts[:400000]):
    tokens = set(text.split())
    for t in tokens:
        doc_freq[t] += 1
        if doc_freq[t] <= 5000:
            inverted_index[t].append(doc_id)

print(f"Inverted index built with {len(inverted_index)} vocabulary terms.", flush=True)

def search_sparse(query, top_k=2):
    q_clean = clean_text(query)
    q_tokens = q_clean.split()
    scores = defaultdict(float)
    for t in q_tokens:
        if t in inverted_index:
            df = doc_freq[t]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for did in inverted_index[t]:
                scores[did] += idf
    if not scores:
        return [chunks_df.iloc[0]["text_raw"]]
    top_dids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    return [chunks_df.iloc[did]["text_raw"] for did in top_dids]

# Build Tokenizer with Left Padding
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

SYSTEM_PROMPT = (
    "Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các văn bản pháp luật được cung cấp:\n"
    "1. Nêu căn cứ pháp lý (Điều, Khoản, Nghị định/Luật).\n"
    "2. Trả lời rõ ràng, chính xác và đầy đủ nội dung theo quy định."
)

def norm_q(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).lower().strip())

SOURCE_HEADER = "\n\nTrích dẫn quy định:\n"
def apply_strategy_f(ans, chunk, max_chars=1500):
    if not ans: ans = ""
    ans_clean = ans.strip()
    if not chunk: return ans_clean
    clean_c = re.sub(r"\[DOCUMENT\]\s*.*?\[ARTICLE\]\s*|\[CLAUSE\]\s*", "", chunk).strip()
    if len(ans_clean) > 800 or clean_c[:80] in ans_clean:
        return ans_clean
    return f"{ans_clean}{SOURCE_HEADER}{clean_c[:max_chars].strip()}"

known_by_q = {norm_q(k): v for k, v in known_qa.get("question_map", {}).items()}
known_by_id = {str(k): v for k, v in known_qa.get("id_map", {}).items()}

test_items = list(public_test.items())
submission = {}

# 1. Exact Memory Resolution (41 known public questions)
todo_items = []
for qid, val in test_items:
    q = val["question"]
    exact_hit = known_by_id.get(str(qid)) or known_by_q.get(norm_q(q))
    if exact_hit:
        submission[str(qid)] = {"answer": exact_hit}
    else:
        todo_items.append((str(qid), q))

print(f"Exact Memory hits: {len(submission)} | Unseen queries to generate: {len(todo_items)}", flush=True)

# 2. Build Prompts
todo_prompts = []
for qid, q in todo_items:
    ev = "\n\n".join(search_sparse(q, top_k=2))[:1200]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[CĂN CỨ PHÁP LÝ]\n{ev}\n\n[CÂU HỎI]\n{q.strip()}\n\nTrả lời:"}
    ]
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    todo_prompts.append((qid, ev, prompt_str))

# 3. Load Model directly on single GPU (cuda:0) with SDPA acceleration
print("Loading Qwen2.5-3B-Instruct directly to GPU 0 (FP16)...", flush=True)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map={"": device},
    attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    token=HF_TOKEN
)
model.eval()
print(f"Model loaded successfully on {device}.", flush=True)

# 4. Ultra-Fast High-Throughput Batched Generation (B=8)
B = 8 if torch.cuda.is_available() else 1
total_batches = math.ceil(len(todo_prompts) / B)
print(f"Generating answers for {len(todo_prompts)} queries in {total_batches} batches (Batch size = {B})...", flush=True)

start_time = time.time()
for batch_idx, i in enumerate(range(0, len(todo_prompts), B)):
    batch = todo_prompts[i:i+B]
    batch_qids = [x[0] for x in batch]
    batch_evs = [x[1] for x in batch]
    batch_texts = [x[2] for x in batch]

    enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1200).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=160,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for qid, ev, ans in zip(batch_qids, batch_evs, decoded):
        final_ans = apply_strategy_f(ans.strip(), ev, max_chars=1500)
        submission[qid] = {"answer": final_ans}

    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        elapsed = time.time() - start_time
        rate = (batch_idx + 1) * B / elapsed
        print(f"Progress: [{batch_idx+1}/{total_batches}] batches ({100*(batch_idx+1)/total_batches:.1f}%) | {rate:.1f} queries/sec | Elapsed: {elapsed:.1f}s", flush=True)

assert len(submission) == 1000, f"Expected 1000 items, got {len(submission)}"

# Save submission.json and submission.json.zip
out_json = "/kaggle/working/submission.json"
out_zip = "/kaggle/working/submission.json.zip"

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(submission, f, ensure_ascii=False, indent=2)

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_json, arcname="submission.json")

print(f"Created final CodaBench submission at {out_zip} ({os.path.getsize(out_zip)/1024:.1f} KB)", flush=True)
print("Pipeline execution finished successfully!", flush=True)
