Here is a comprehensive README.md file for your repository, written based on your project details and specifically explaining your use of MUVERA to reduce latency.

-----

# RAG System for NCIIPC AI Grand Challenge

This repository contains an implementation of a high-performance retrieval system for the [NCIIPC-Startup India AI Grand Challenge](https://ai-grand-challenge.in/) on "Retrieval Augmented Generation based Question and Answering System."

The solution focuses on the **retrieval** component of RAG, implementing a state-of-the-art **two-stage retrieve-and-rerank pipeline** using MUVERA (FDE) and ColBERT to achieve both high speed and high accuracy on a large document corpus.

-----

## 🎯 The Challenge

The challenge requires a RAG solution to sift through a large volume of unstructured documents, correlate information, and provide fact-grounded, traceable answers to user queries.

### Key Objectives

1.  **Relevant documents retrieval** from a diverse documents corpus.
2.  **Generate fluent, factually grounded** and explainable natural language responses.
3.  **Minimize hallucination** & ensure all claims are traceable to source content.

> **Note:** This project implements a state-of-the-art solution for **Objective 1: Relevant documents retrieval**. This high-performance retrieval system is the essential foundation upon which a generative model (Objective 2) would be built.

-----

## 💡 Our Approach: A Two-Stage Retrieve-and-Rerank Pipeline

To efficiently search a "humongous" corpus, a single-pass system is often either too slow (if accurate) or too inaccurate (if fast). We implemented a **two-stage retrieve-and-rerank pipeline** to get the best of both worlds.

1.  **Stage 1: Fast Retrieval (MUVERA FDE)**

      * **Goal:** To scan the *entire* corpus at extremely high speed.
      * **Method:** We use the principles from the [muvera-py](https://github.com/sionic-ai/muvera-py) repository to create **Fixed-Dimensional Encodings (FDE)**. This compresses the complex, multi-vector ColBERT representation of *each document chunk* into a **single, compact vector**.
      * **Result:** The search becomes a blazing-fast matrix multiplication (a dot product) that instantly finds a set of promising candidates (e.g., the top 100 chunks).

2.  **Stage 2: High-Quality Reranking (Native ColBERT)**

      * **Goal:** To deeply analyze the top 100 candidates and find the *best* matches.
      * **Method:** We use the full, powerful **native ColBERT `maxSim` operation** (late interaction) to re-score *only* this small subset of candidates.
      * **Result:** We get a highly accurate and relevant final ranking, having avoided the cost of running the expensive ColBERT model on the entire dataset.

-----

## 🔧 How It Works

### 1\. Advanced Data Processing (`data_processing.py`)

The provided mock dataset consisted of `.txt` files containing a mix of plain text and semi-structured, JSON-like content. A custom data processing pipeline was built to:

  * Intelligently parse both text formats.
  * Clean and normalize the text.
  * Chunk all documents into smaller, semantically coherent pieces (e.g., `doc_9683_chunk1`).
  * Create a final `corpus.json` file that maps every chunk ID back to its parent document (e.g., `doc_9683.txt`), which is crucial for the final submission.

### 2\. The MUVERA + ColBERT Pipeline (`fde_colbert_v2.py`)

This script contains the core logic in the `ColbertFdeRetriever` class:

1.  **Indexing:**

      * The entire (chunked) corpus is loaded.
      * For each chunk, we generate its full, high-dimensional **ColBERT multi-vector embeddings** and store them in a dictionary (`doc_embeddings_map`).
      * Simultaneously, we use the MUVERA **Fixed-Dimensional Encoding (FDE)** logic (`generate_document_fde_batch`) to compress these multi-vector embeddings into a single-vector representation for each chunk.
      * These FDE vectors are stacked into a single large matrix (`self.fde_index`), which is our fast-search index.

2.  **Searching (Stage 1 - Retrieve):**

      * A user's query is received.
      * The query is encoded into its FDE vector representation.
      * The search is executed as a **single matrix multiplication** (`self.fde_index @ query_fde`) between the query vector and the entire document index.
      * This instantly returns a list of candidate chunks, sorted by their approximate FDE score.

3.  **Reranking (Stage 2 - Rerank):**

      * We take the top `k` (e.g., 40) candidates from the FDE search.
      * We retrieve their **full ColBERT embeddings** (which we saved during indexing) from the `doc_embeddings_map`.
      * We run the computationally expensive, high-accuracy `self.ranker()` (native ColBERT `maxSim`) on *only* these 40 candidates.
      * This produces the final, highly relevant list of chunks, sorted by their true ColBERT score.

-----

## 🚀 How MUVERA (FDE) Reduces Latency

A native ColBERT search is powerful but slow. It requires a "late-interaction" `maxSim` operation, which compares every vector in a query against every vector in *every single document* in the corpus. This is computationally infeasible for millions of documents in a real-time setting.

Our solution, based on the **MUVERA** paper, solves this exact latency problem:

1.  **From Multi-Vector to Single Vector:** MUVERA's FDE technique (which we implement in `fde_generator.py`) compresses the thousands of vectors in a document's multi-vector representation into a **single, fixed-dimensional vector**.
2.  **From `maxSim` to Dot Product:** This compression cleverly approximates the expensive `maxSim` operation. Instead of a complex, token-by-token comparison, our initial search becomes a **simple, hardware-accelerated dot product (matrix multiplication)**. This is *orders of magnitude faster* than native ColBERT.
3.  **A "Filter," Not a Replacement:** We don't rely on the FDE score for the final answer. We use it as a massive "filter" to instantly **reduce a million-document problem into a 100-document problem**. The expensive, high-accuracy ColBERT reranking is then easily applied to this tiny subset, giving us a final system that is both fast (low-latency) and accurate.

```bash
python prepare_submission.py
# Output: This will create 'PS04_TEAM_NAME.zip'
```
