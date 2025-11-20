import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import bm25s
import jsonlines
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import Stemmer
from sentence_transformers import CrossEncoder


clapnq_CORPUS_FILE = "./docs/clapnq.jsonl"
clapnq_QUERY_FILE = "./docs/clapnq_rewrite.jsonl"
cloud_CORPUS_FILE = "./docs/cloud.jsonl"
cloud_QUERY_FILE = "./docs/cloud_rewrite.jsonl"
fiqa_CORPUS_FILE = "./docs/fiqa.jsonl"
fiqa_QUERY_FILE = "./docs/fiqa_rewrite.jsonl"
govt_CORPUS_FILE = "./docs/govt.jsonl"
govt_QUERY_FILE = "./docs/govt_rewrite.jsonl"
BM25_OUTPUT_FILE = "./results/bm25_results.jsonl"
FAISS_OUTPUT_FILE = "./results/faiss_results.jsonl"
HYBRID_OUTPUT_FILE = "./results/hybrid_rerank_results.jsonl"
TOP_K = 10
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# load passages 
def load_passages(CORPUS_FILE):
    passages = []
    passage_ids = []

    with jsonlines.open(CORPUS_FILE, "r") as reader:
        for item in reader:
            text = item.get("text") or item.get("content")
            doc_id = item.get("_id") or item.get("document_id")
            passages.append(text)
            passage_ids.append(doc_id)

    print(f"Loaded {len(passages)} passages")
    return passages, passage_ids

# load queries
def parse_queries(QUERY_FILE):
    raw_queries = []
    with jsonlines.open(QUERY_FILE, "r") as reader:
        for item in reader:
            raw_queries.append(item)

    # Parse queries into conversation_id, task_id, and text
    queries_parsed = []
    for q in raw_queries:
        _id = q["_id"]
        text = q["text"].strip()

        if "<::>" in _id:
            conv_id, turn_num = _id.split("<::>")
        else:
            conv_id, turn_num = _id, "0"

        queries_parsed.append({
            "conversation_id": conv_id,
            "task_id": _id,
            "text": text
        })

    print("Example parsed query:", queries_parsed[0])
    print("Total queries:", len(queries_parsed))
    return queries_parsed


clapnq_passages, clapnq_passage_ids = load_passages(clapnq_CORPUS_FILE)
cloud_passages, cloud_passage_ids = load_passages(cloud_CORPUS_FILE)
fiqa_passages, fiqa_passage_ids = load_passages(fiqa_CORPUS_FILE)
govt_passages, govt_passage_ids = load_passages(govt_CORPUS_FILE)

clapnq_queries = parse_queries(clapnq_QUERY_FILE)
cloud_queries = parse_queries(cloud_QUERY_FILE)
fiqa_queries = parse_queries(fiqa_QUERY_FILE)
govt_queries = parse_queries(govt_QUERY_FILE)

ALPHA = 0.6

def hybrid_retrieval(passage_ids, passages, queries, name, alpha=ALPHA, model_name=MODEL_NAME, batch_size=1024):

    print(f"\nRunning hybrid retrieval for {name} (alpha={alpha})")

    # Sparse - BM25
    print("Building BM25 index...")
    stemmer = Stemmer.Stemmer("en")
    tokenized_corpus = bm25s.tokenize(passages, stopwords="en", stemmer=stemmer)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus)

    # Dense - BAAI/bge-base-en-v1.5
    print("Loading dense model:", model_name)
    dense_model = SentenceTransformer(model_name, device=device)

    # Reranker - BAAI/bge-reranker-v2-m3
    reranker_model_name = "BAAI/bge-reranker-v2-m3"
    print("Loading reranker:", reranker_model_name)
    reranker = CrossEncoder(reranker_model_name, device=device)

    # Encoding passages
    print("Encoding passages with SentenceTransformer...")
    passage_emb_list = []
    for start in tqdm(range(0, len(passages), batch_size), desc="Passage batches"):
        batch_texts = passages[start:start + batch_size]
        with torch.no_grad():
            emb = dense_model.encode(
                batch_texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )
        passage_emb_list.append(emb)

    passage_emb = torch.cat(passage_emb_list, dim=0).cpu().float().numpy()
    faiss.normalize_L2(passage_emb)

    dim = passage_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_emb)

    # Encoding Queries
    print("Encoding queries with SentenceTransformer...")
    dense_queries_list = []
    query_texts = [q["text"] for q in queries]

    for start in tqdm(range(0, len(query_texts), batch_size), desc="Query batches"):
        batch_texts = query_texts[start:start + batch_size]
        with torch.no_grad():
            emb = dense_model.encode(
                batch_texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )
        dense_queries_list.append(emb)

    dense_queries = torch.cat(dense_queries_list, dim=0).cpu().float().numpy()
    faiss.normalize_L2(dense_queries)

    # Hybrid Retrieval + Reranking
    with jsonlines.open(HYBRID_OUTPUT_FILE, mode="a") as writer:
        for qi, q in enumerate(tqdm(queries, desc="Combining results")):
            q_text = q["text"]

            # Sparse results (BM25) - top_50
            q_tokens = bm25s.tokenize(q_text, stemmer=stemmer)
            s_results, s_scores = bm25.retrieve(
                q_tokens,
                k=5 * TOP_K,
                corpus=passages,
            )

            # Dense results (FAISS) - top_50
            q_emb = dense_queries[qi].reshape(1, -1)
            d_scores, d_idx = index.search(q_emb, 5 * TOP_K)

            # Normalize scores - [0,1]
            s_scores = (s_scores - np.min(s_scores)) / (np.max(s_scores) - np.min(s_scores) + 1e-9)
            d_scores = (d_scores - np.min(d_scores)) / (np.max(d_scores) - np.min(d_scores) + 1e-9)

            combined_scores = {}

            # Add sparse (BM25) scores
            for passage_text, s_score in zip(s_results[0], s_scores[0]):
                pid = passage_ids[passages.index(passage_text)]
                combined_scores[pid] = float((1 - alpha) * s_score)

            # Add dense (FAISS) scores
            for d_score, idx in zip(d_scores[0], d_idx[0]):
                pid = passage_ids[idx]
                combined_scores[pid] = combined_scores.get(pid, 0.0) + float(alpha * d_score)

            # --- Select top 50 hybrid candidates ---
            top_candidates = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5 * TOP_K]

            candidate_pids = [pid for pid, _ in top_candidates]
            candidate_texts = [passages[passage_ids.index(pid)] for pid in candidate_pids]

            # Rerank the top 50 and select top 10
            pairs = [(q_text, passage_text) for passage_text in candidate_texts]
            ce_scores = reranker.predict(pairs)
            reranked_idx = np.argsort(ce_scores)[::-1][:TOP_K]

            context_list = []
            for idx in reranked_idx:
                pid = candidate_pids[idx]
                score = float(ce_scores[idx])
                context_list.append({"document_id": pid, "score": score})

            # write to the results file
            writer.write(
                {
                    "conversation_id": q["conversation_id"],
                    "task_id": q["task_id"],
                    "contexts": context_list,
                    "Collection": f"mt-rag-{name}",
                }
            )

    print(f"Hybrid results saved to {HYBRID_OUTPUT_FILE}")



hybrid_retrieval(clapnq_passage_ids, clapnq_passages, clapnq_queries, "clapnq")
hybrid_retrieval(cloud_passage_ids, cloud_passages, cloud_queries, "cloud")
hybrid_retrieval(fiqa_passage_ids, fiqa_passages, fiqa_queries, "fiqa")
hybrid_retrieval(govt_passage_ids, govt_passages, govt_queries, "govt")