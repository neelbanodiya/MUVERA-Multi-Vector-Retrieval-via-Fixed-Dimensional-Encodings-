import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from dataclasses import replace

# Import third-party libraries
from neural_cherche import models, rank
from ir_datasets.util import Cache

# Import our custom modules
from fde_generator import (
    FixedDimensionalEncodingConfig,
    generate_document_fde_batch,
    generate_query_fde,
    EncodingType,
    ProjectionType,
)



MUVERA_PY_PATH = "./muvera-py"
if not os.path.isdir(MUVERA_PY_PATH):
    raise FileNotFoundError(f"The directory '{MUVERA_PY_PATH}' was not found. Please clone the repository first.")

# Change current directory
os.chdir(MUVERA_PY_PATH)


class ColbertFdeRetriever:
    """
    Implements a two-stage retrieve-and-rerank pipeline.
    1. Fast retrieval using FDE (MUVERA).
    2. Accurate reranking using native ColBERT on the retrieved candidates.
    """
    def __init__(self, model_name="colbert-ir/colbertv2.0"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.ColBERT(model_name_or_path=model_name, device=device)
        self.ranker = rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=5,
            projection_type=ProjectionType.AMS_SKETCH,
            projection_dimension=16,
            seed=42,
            encoding_type=EncodingType.AVERAGE,
            fill_empty_partitions=True,
        )

        self.fde_index = None
        self.doc_ids = []
        self.doc_embeddings_map = {} # Will store {doc_id: embedding}
        print("ColbertFdeRetriever initialized successfully.")

    def index(self, corpus: dict):
        print("\n--- Starting Corpus Indexing ---")
        start_time = time.time()
        #Sorting for consistent results
        self.doc_ids = sorted(list(corpus.keys()))
        documents_for_ranker = [{"id": doc_id, **corpus[doc_id]} for doc_id in self.doc_ids]

        print(f"Generating native multi-vector embeddings for {len(documents_for_ranker)} documents...")
        self.doc_embeddings_map = self.ranker.encode_documents(
            documents=documents_for_ranker, batch_size=32
        )
        doc_embeddings_list = [self.doc_embeddings_map[doc_id] for doc_id in self.doc_ids]

        print(f"Multi-vector embedding generation took {time.time() - start_time:.2f} seconds.")

        print("Generating FDEs from ColBERT embeddings using the optimized BATCH function...")
        start_fde_time = time.time()
        self.fde_index = generate_document_fde_batch(
            doc_embeddings_list, self.doc_config
        )
        print(f"FDE generation took {time.time() - start_fde_time:.2f} seconds.")
        print(f"--- Corpus Indexing Finished in {time.time() - start_time:.2f} seconds ---")

    def search_fde(self, query: str, top_k: int = 10) -> dict:
        """First stage: Fast retrieval using FDE."""
        query_embeddings_map = self.ranker.encode_queries(queries=[query])
        query_embeddings = list(query_embeddings_map.values())[0]
        query_config = replace(self.doc_config, fill_empty_partitions=False)
        query_fde = generate_query_fde(query_embeddings, query_config)
        scores = self.fde_index @ query_fde

        sorted_results = sorted(zip(self.doc_ids, scores), key=lambda item: item[1], reverse=True)
        top_k_results = sorted_results[:top_k]
        return {doc_id: float(score) for doc_id, score in top_k_results}

    def rerank_with_colbert(self, query: str, candidate_doc_ids: List[str]) -> Dict[str, float]:
        """Second stage: Rerank candidate documents using native ColBERT."""
        print(f"\n--- Reranking {len(candidate_doc_ids)} documents with native ColBERT ---")

        # Encode the query once
        query_embeddings = self.ranker.encode_queries(queries=[query])

        # This part is correct: A dictionary of the necessary document embeddings
        rerank_doc_embeddings = {
            doc_id: self.doc_embeddings_map[doc_id] for doc_id in candidate_doc_ids
        }

        # **THE FINAL FIX IS HERE:**
        # The `documents` parameter must be a list of dictionaries,
        # where each dictionary contains the key for the document ID.
        documents_for_reranking = [{"id": doc_id} for doc_id in candidate_doc_ids]

        # The ranker can now correctly look up document['id']
        reranked_results = self.ranker(
            queries_embeddings=query_embeddings,
            documents_embeddings=rerank_doc_embeddings,
            documents=[documents_for_reranking] # This now needs to be wrapped in a list for each query
        )

        # The output is a list of lists, so we take the first element
        return {item["id"]: item["similarity"] for item in reranked_results[0]}

def process_queries_in_batch(
    retriever: ColbertFdeRetriever,
    queries: List[Dict[str, Any]],
    output_file_path: str,
    batch_size: int = 25,
    top_k_retrieval: int = 40,
    top_k_final: int = 10 , # <-- New parameter to control final output size
) -> None:
    """
    Processes queries in batches, performs retrieve-and-rerank, and saves the top_k_final
    results to a JSONL file.
    """
    print(f"\n--- Starting batch processing of {len(queries)} queries ---")
    print(f"Batch size: {batch_size}, Top-K retrieval: {top_k_retrieval}, Final Top-K: {top_k_final}")
    print(f"Results will be saved to: {output_file_path}")

    with open(output_file_path, "w") as out_file:
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing Query Batches"):
            batch = queries[i : i + batch_size]
            batch_query_texts = [q["query"] for q in batch]

            batch_candidates = [
                retriever.search_fde(query=q_text, top_k=top_k_retrieval)
                for q_text in batch_query_texts
            ]

            for idx, query_info in enumerate(batch):
                candidate_ids = list(batch_candidates[idx].keys())

                if not candidate_ids:
                    final_reranked_scores = {}
                else:
                    final_reranked_scores = retriever.rerank_with_colbert(
                        query=query_info["query"], candidate_doc_ids=candidate_ids
                    )
                
                # --- CHANGE IS HERE ---
                # 1. Sort the final reranked scores by score, descending
                sorted_reranked_results = sorted(
                    final_reranked_scores.items(), key=lambda item: item[1], reverse=True
                )
                
                # 2. Take only the top_k_final results
                top_k_final_results = dict(sorted_reranked_results[:top_k_final])
                # --- END OF CHANGE ---

                output_record = {
                    "query_num": query_info["query_num"],
                    "query": query_info["query"],
                    "results": top_k_final_results, # Use the sliced results
                }

                # This is the correct logic for JSONL: one JSON object, one newline.
                out_file.write(json.dumps(output_record) + "\n")

    print(f"\n--- Batch processing complete. Results saved to {output_file_path} ---")


if __name__ == '__main__':
    # --- Configuration ---
    CORPUS_FILE_PATH = "path/to/your/corpus.json"  # <-- CHANGE THIS
    QUERIES_FILE_PATH = "path/to/your/queries.json" # <-- CHANGE THIS
    RESULTS_OUTPUT_PATH = "retrieval_results.jsonl"
    QUERY_BATCH_SIZE = 25 # Adjust based on your GPU memory
    TOP_K_CANDIDATES = 40 # How many documents to retrieve before reranking

    # --- Step 1: Load Your Real Corpus and Queries ---
    print("Loading corpus and queries...")
    with open(CORPUS_FILE_PATH, "r") as f:
        corpus = json.load(f)

    with open(QUERIES_FILE_PATH, "r") as f:
        queries = json.load(f)
    
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries.")

    # --- Step 2: Initialize and Index the Corpus ---
    # This step can take a while for large corpora
    fde_retriever = ColbertFdeRetriever()
    fde_retriever.index(corpus=corpus)
    
    # --- Step 3: Run the Batch Processing Pipeline ---
    process_queries_in_batch(
        retriever=fde_retriever,
        queries=queries,
        output_file_path=RESULTS_OUTPUT_PATH,
        batch_size=QUERY_BATCH_SIZE,
        top_k_retrieval=TOP_K_CANDIDATES,
    )