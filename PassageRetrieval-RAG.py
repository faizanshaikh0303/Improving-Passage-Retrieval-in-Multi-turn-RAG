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
import daft
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import math

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
HYBRID_OUTPUT_FILE = "./results/hybrid_results.jsonl"
TOP_K = 10
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

      # Split _id into conversation_id and turn number
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


def sparse_retrieval(passage_ids, passages, queries_parsed, name, output_file=BM25_OUTPUT_FILE):
    print(f"\nRunning BM25S retrieval for {name}...")
    stemmer = Stemmer.Stemmer('en')
    tokenized_corpus = bm25s.tokenize(passages, stopwords="en",stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    # Map passage text -> document ID
    passage2id = {p: pid for p, pid in zip(passages, passage_ids)}

    with jsonlines.open(output_file, mode='a') as writer:
        for q in tqdm(queries_parsed):
            query_tokens = bm25s.tokenize(q["text"],stemmer=stemmer)
            results, scores = retriever.retrieve(query_tokens, k=TOP_K, corpus=passages)

            context_list = []
            for j in range(TOP_K):
                passage_text = results[0, j]
                doc_id = passage2id[passage_text]
                context_list.append({
                    "document_id": doc_id,
                    "score": float(scores[0, j])
                })

            writer.write({
                "conversation_id": q["conversation_id"],
                "task_id": q["task_id"],
                "contexts": context_list,
                "Collection": f"mt-rag-{name}"
            })

    print(f"BM25S results appended to {output_file}")


def dense_retrieval(passage_ids, passages, queries_parsed, name,
                                       output_file=FAISS_OUTPUT_FILE,
                                       model_name=MODEL_NAME,
                                       batch_size=4096,
                                       use_history=True):
    """
    Dense retrieval pipeline with Daft-based query reformulation.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    query_texts=[]
    for q in queries_parsed:
        query_texts.append(q["text"])

    # query_texts = df_queries["reformulated_text"].tolist()

    #Load embedding model
    model = SentenceTransformer(model_name, device=device).half()

    #Encode passages using Daft batching
    print("Encoding passages...")
    df_passages = daft.from_pandas(pd.DataFrame({"passage_id": passage_ids, "text": passages}))
    passage_embeddings_list = []

    for batch in tqdm(df_passages.to_arrow_iter(), desc="Passage Batches"):
        texts = batch["text"].to_pylist()
        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
        passage_embeddings_list.append(embeddings)

    passage_embeddings = torch.cat(passage_embeddings_list, dim=0).cpu().float().numpy()
    faiss.normalize_L2(passage_embeddings)

    #Build FAISS index
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)
    print(f"FAISS index built with {len(passage_embeddings)} passages.")

    #Encode queries using Daft batching
    print("Encoding queries...")
    df_queries_daft = daft.from_pandas(pd.DataFrame({"text": query_texts}))
    query_embeddings_list = []

    for batch in tqdm(df_queries_daft.to_arrow_iter(), desc="Query Batches"):
        texts = batch["text"].to_pylist()
        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
        query_embeddings_list.append(embeddings)

    query_embeddings = torch.cat(query_embeddings_list, dim=0).cpu().float().numpy()
    faiss.normalize_L2(query_embeddings)

    #Retrieve top-K passages
    print("Retrieving top-K passages...")
    with jsonlines.open(output_file, mode='a') as writer:
        for i, query_emb in enumerate(tqdm(query_embeddings, desc="Queries")):
            query_emb = query_emb.reshape(1, -1)
            D, I = index.search(query_emb, TOP_K)
            context_list = [{"document_id": passage_ids[idx], "score": float(score)}
                            for score, idx in zip(D[0], I[0])]

            writer.write({
                "conversation_id": queries_parsed[i]["conversation_id"],
                "task_id": queries_parsed[i]["task_id"],
                "contexts": context_list,
                "Collection": f"mt-rag-{name}"
            })

    print(f"FAISS results appended to {output_file}")


clapnq_passages, clapnq_passage_ids = load_passages(clapnq_CORPUS_FILE)
cloud_passages, cloud_passage_ids = load_passages(cloud_CORPUS_FILE)
fiqa_passages, fiqa_passage_ids = load_passages(fiqa_CORPUS_FILE)
govt_passages, govt_passage_ids = load_passages(govt_CORPUS_FILE)

clapnq_queries = parse_queries(clapnq_QUERY_FILE)
cloud_queries = parse_queries(cloud_QUERY_FILE)
fiqa_queries = parse_queries(fiqa_QUERY_FILE)
govt_queries = parse_queries(govt_QUERY_FILE)

# sparse_retrieval(clapnq_passage_ids,clapnq_passages,clapnq_queries,"clapnq")
# sparse_retrieval(cloud_passage_ids,cloud_passages,cloud_queries,"cloud")
# sparse_retrieval(fiqa_passage_ids,fiqa_passages,fiqa_queries,"fiqa")
# sparse_retrieval(govt_passage_ids,govt_passages,govt_queries,"govt")

# dense_retrieval(clapnq_passage_ids, clapnq_passages, clapnq_queries, "clapnq")
# dense_retrieval(cloud_passage_ids, cloud_passages, cloud_queries, "cloud")
# dense_retrieval(fiqa_passage_ids, fiqa_passages, fiqa_queries, "fiqa")
# dense_retrieval(govt_passage_ids, govt_passages, govt_queries, "govt")

ALPHA = 0.6
def hybrid_retrieval(passage_ids, passages, queries, name, alpha=ALPHA, model_name=MODEL_NAME):
    print(f"\nRunning hybrid retrieval for {name} (alpha={alpha})")

    # --- Sparse: BM25 ---
    print("Building BM25 index...")
    stemmer = Stemmer.Stemmer('en')
    tokenized_corpus = bm25s.tokenize(passages, stopwords="en", stemmer=stemmer)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus)

    # --- Dense: SentenceTransformer + FAISS ---
    print("Encoding passages with SentenceTransformer...")
    dense_model = SentenceTransformer(model_name, device=device)
    df_passages = daft.from_pandas(pd.DataFrame({"text": passages}))
    passage_emb_list = []
    for batch in tqdm(df_passages.to_arrow_iter(), desc="Passage batches"):
        texts = batch["text"].to_pylist()
        with torch.no_grad():
            emb = dense_model.encode(texts, convert_to_tensor=True, device=device)
        passage_emb_list.append(emb)
    passage_emb = torch.cat(passage_emb_list, dim=0).cpu().float().numpy()
    faiss.normalize_L2(passage_emb)

    dim = passage_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_emb)

    # --- Process Queries ---
    print("Encoding queries and retrieving...")
    dense_queries = []
    df_queries = daft.from_pandas(pd.DataFrame({"text": [q["text"] for q in queries]}))
    for batch in tqdm(df_queries.to_arrow_iter(), desc="Query batches"):
        texts = batch["text"].to_pylist()
        with torch.no_grad():
            emb = dense_model.encode(texts, convert_to_tensor=True, device=device)
        dense_queries.append(emb)
    dense_queries = torch.cat(dense_queries, dim=0).cpu().float().numpy()
    faiss.normalize_L2(dense_queries)

    
    with jsonlines.open(HYBRID_OUTPUT_FILE, mode='a') as writer:
        for qi, q in enumerate(tqdm(queries, desc="Combining results")):
            # Sparse results
            q_tokens = bm25s.tokenize(q["text"], stemmer=stemmer)
            s_results, s_scores = bm25.retrieve(q_tokens, k=2*TOP_K, corpus=passages)

            # Dense results
            q_emb = dense_queries[qi].reshape(1, -1)
            d_scores, d_idx = index.search(q_emb, 2*TOP_K)

            # Normalize both scores
            s_scores = (s_scores - np.min(s_scores)) / (np.max(s_scores) - np.min(s_scores) + 1e-9)
            d_scores = (d_scores - np.min(d_scores)) / (np.max(d_scores) - np.min(d_scores) + 1e-9)
            # Combine scores
            combined_scores = {}

            # Add sparse (BM25) scores
            for passage_text, s_score in zip(s_results[0], s_scores[0]):
                pid = passage_ids[passages.index(passage_text)]
                combined_scores[pid] = float((1 - alpha) * s_score)

            # Add dense (FAISS) scores
            for d_score, idx in zip(d_scores[0], d_idx[0]):
                pid = passage_ids[idx]
                combined_scores[pid] = combined_scores.get(pid, 0.0) + float(alpha * d_score)

            # Sort by hybrid score
            ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            context_list = [{"document_id": pid, "score": float(score)} for pid, score in ranked]

            writer.write({
                "conversation_id": q["conversation_id"],
                "task_id": q["task_id"],
                "contexts": context_list,
                "Collection": f"mt-rag-{name}-hybrid"
            })

    print(f"Hybrid results saved to {HYBRID_OUTPUT_FILE}")

hybrid_retrieval(clapnq_passage_ids, clapnq_passages, clapnq_queries, "clapnq")
hybrid_retrieval(cloud_passage_ids, cloud_passages, cloud_queries, "cloud")
hybrid_retrieval(fiqa_passage_ids, fiqa_passages, fiqa_queries, "fiqa")
hybrid_retrieval(govt_passage_ids, govt_passages, govt_queries, "govt")