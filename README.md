# Improving-Passage-Retrieval-in-Multi-turn-RAG
Report for PA3: Improving Passage Retrieval in Multi-turn RAG
Name: Faizan Shaikh

# Methodology:
The code takes in corpus and query files in JSONL format. Corpus/Queries are parsed to extract id and text. The text is cleaned and prepared for tokenization.

Sparse Retrieval (BM25)

Tokenization: Input text is tokenized using an English stemmer to reduce word variations to their root forms.
Indexing: An inverted index is built on the entire corpus for efficient keyword matching.
Retrieval: For each query, the top 50 documents are retrieved based on BM25 scores.
Advantages: fast and interpretable; good for keyword-based queries.
Limitations: cannot capture semantic similarity beyond exact or stemmed token overlap.

Dense Retrieval (SentenceTransformer + FAISS)

Encoder: The BAAI/bge-base-en-v1.5 model is used to encode both passages and queries into high-dimensional vectors.
Indexing: Embeddings are normalized using L2 normalization and indexed using FAISS (Facebook AI Similarity Search) with a Flat Inner Product index (IndexFlatIP) to calculate cosine similarity.
Retrieval: Similar to the sparse stream, the top 50 documents are retrieved based on vector similarity.
Advantages: captures semantic similarity; handles synonyms and paraphrasing.
Limitations: embedding computation is GPU intensive; requires sufficient memory for large corpora. This was performed on a colab GPU.

Combining (Sparse + Dense)

Normalization: Scores from BM25 and FAISS are Min-Max normalized to ensure they share a common scale [0, 1].
Weighting: The final hybrid score is calculated using an alpha parameter (𝜶 = 0.6), prioritizing dense retrieval slightly higher than sparse:
                                        Shybrid= (1-𝜶)(Ssparse) + (𝜶)(Sdense)
Candidate Selection: The top 50 candidates with the highest hybrid scores are passed to the reranker.

Reranking (BAAI/bge-reranker-v2-m3)

The final stage utilizes a Cross-Encoder model, specifically BAAI/bge-reranker-v2-m3. 
Unlike the bi-encoder used in the dense stage (which encodes query and document separately), the Cross-Encoder processes the query and document simultaneously, allowing it to capture fine-grained interactions between queries and passage tokens. 
The top K=10 documents are selected based on these re-calculated scores.

# Experiments and Results:
Datasets
CLAPNQ: 183408 passages 208 queries
CLOUD: 61022 passages 188 queries
FIQA: 49607 passages 180 queries
GOVT: 72422 passages 201 queries

Evaluation Metrics

Recall@K: Fraction of relevant documents retrieved in top-K results.
nDCG@K: Normalized Discounted Cumulative Gain, measuring ranking quality.

Results by Collection

Collection   R@1     R@3    R@5   R@10   nDCG@1   nDCG@3   nDCG@5   nDCG@10

clapnq      0.219   0.431  0.522  0.656  0.524    0.478    0.503     0.560

cloud       0.203   0.350  0.398  0.487  0.404    0.364    0.379     0.420

fiqa        0.156   0.297  0.371  0.486  0.350    0.320    0.348     0.396

govt        0.186   0.383  0.481  0.603  0.408    0.399    0.437     0.488

# Average (Across Collections)

                   @1     @3     @5    @10
Baseline Recall   0.08   0.15   0.20   0.27
Recall            0.192  0.368  0.446  0.562
nDCG              0.425  0.393  0.420  0.469


# Analysis and Discussion:

Performance Gains

The implemented method yields a 108.1% increase in Recall@10 (0.27 to 0.562) and a 123.3% increase in nDCG@10 (0.21 to 0.469) compared to the BM25 baseline.
This substantial improvement validates the importance of the multi-stage pipeline.

Synergistic Retrieval

The hybrid approach removes the individual weaknesses of sparse and dense retrieval. 
BM25 ensures that documents containing specific entity names or rare keywords are not missed (high precision), while the BGE embedding model captures conceptual relevance even when vocabulary mismatches occur (high recall). 
The fusion parameter 𝜶=0.6 was chosen because, while semantic similarity is the primary driver, lexical matching remains critical.

Impact of Reranking

The inclusion of the bge-reranker-v2-m3 is the reason for the high nDCG scores.
By re-evaluating the top 50 candidates using a computationally heavier but more accurate Cross-Encoder, the code effectively filters out "false positives" - documents that are vector-space neighbors but not truly relevant to the query's intent. 
This ensures that the top-ranked documents (Top-1 to Top-3) are highly relevant, as evidenced by the strong nDCG@1 scores (e.g., 0.52 for ClapNQ).


# Discussion and Conclusion

Encoding all passages in batches leverages colab GPU efficiently.
FAISS allows fast similarity search even on large corpora.
Fine-tuning embeddings on domain-specific data may improve FIQA and GOVT embeddings for better recall..
Experimenting with different α values can optimize the hybrid results.
The implemented hybrid retrieval system effectively combines sparse and dense retrieval with cross-encoder reranking, achieving strong performance across multiple collections. Weighted averaging of BM25 and dense scores, followed by reranking, demonstrates the power of multi-stage retrieval pipelines for diverse query sets.
