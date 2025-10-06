import time
import torch
import numpy as np
import json
from tqdm import tqdm
from dataclasses import replace

# Import your existing classes and functions
# Make sure these files (fde_generator.py, etc.) are in the same directory
from neural_cherche import models, rank
from fde_generator import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    EncodingType,
    ProjectionType,
)

# Your ColbertFdeRetriever class from the notebook
# I've included it here for completeness.
class ColbertFdeRetriever:
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
        self.fde_index, self.doc_ids = None, []
        print("ColbertFdeRetriever initialized.")

    def load_index(self, index_path="fde_index.npy", doc_ids_path="doc_ids.json"):
        """Loads the pre-computed index and doc IDs from files."""
        print("Loading FDE index and document IDs...")
        self.fde_index = np.load(index_path)
        with open(doc_ids_path, "r") as f:
            self.doc_ids = json.load(f)
        print(f"Index loaded successfully. Shape: {self.fde_index.shape}")

    def search(self, query: str, top_k: int = 10) -> dict:
        query_embeddings_map = self.ranker.encode_queries(queries=[query])
        query_embeddings = list(query_embeddings_map.values())[0]
        query_config = replace(self.doc_config, fill_empty_partitions=False)
        query_fde = generate_query_fde(query_embeddings, query_config)
        scores = self.fde_index @ query_fde
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        top_doc_ids = [self.doc_ids[i] for i in top_k_indices]
        top_scores = scores[top_k_indices]
        return {doc_id: float(score) for doc_id, score in zip(top_doc_ids, top_scores)}

def main():
    # --- Configuration ---
    BATCH_SIZE = 250  # Adjust this based on your instance's RAM. Start with 100-500.
    TOP_K_RESULTS = 10
    QUERIES_FILE = "Queries.json"  # Your file with the 41,000 queries
    OUTPUT_FILE = "retrieval_results.jsonl" # The output file

    # --- Step 1: Initialize Retriever and Load Index ---
    retriever = ColbertFdeRetriever()
    retriever.load_index()

    # --- Step 2: Load Queries ---
    print(f"Loading queries from {QUERIES_FILE}...")
    with open(QUERIES_FILE, 'r') as f:
        queries_data = json.load(f)
    
    # Assuming queries_data is a list of {'query_id': 'x', 'query': 'y'}
    # Or a dict like {'query_id': 'query_text'}. We'll convert to a list of tuples.
    if isinstance(queries_data, dict):
        queries = list(queries_data.items())
    else:
        queries = [(item['query_id'], item['query']) for item in queries_data]
    
    num_queries = len(queries)
    print(f"Found {num_queries} queries to process.")

    # --- Step 3: Batch Processing ---
    # Erase the output file if it exists
    with open(OUTPUT_FILE, "w") as f:
        pass # Create an empty file

    # Process queries in batches
    for i in tqdm(range(0, num_queries, BATCH_SIZE), desc="Processing Batches"):
        batch_queries = queries[i:i + BATCH_SIZE]
        batch_results = []

        for query_id, query_text in batch_queries:
            results = retriever.search(query_text, top_k=TOP_K_RESULTS)
            batch_results.append({"query_id": query_id, "results": results})

        # Append the results of the entire batch to the file
        with open(OUTPUT_FILE, "a") as f:
            for item in batch_results:
                f.write(json.dumps(item) + "\\n")

    print(f"\\nProcessing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()