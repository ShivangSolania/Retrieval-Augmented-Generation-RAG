"""
Retrieval Quality Evaluation Script
====================================

Evaluates the RAG pipeline's retrieval quality using a golden query set.
Computes Precision@k, MRR, NDCG@k, and recall metrics.

Usage:
    python evaluation/evaluate.py --golden-set evaluation/golden_queries.json \
                                  --doc-id test-doc --top-k 10

Outputs:
    - Metrics summary (stdout)
    - Detailed results (evaluation/results.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# Adjust imports based on your package structure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import RAGPipeline


# =========================================================================
# Metrics Computation
# =========================================================================


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@k.
    
    Args:
        retrieved_ids: List of retrieved node IDs (in rank order).
        relevant_ids: Set of relevant node IDs for this query.
        k: Cutoff rank.
    
    Returns:
        Precision@k in range [0, 1].
    """
    if k == 0 or len(retrieved_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_ids: List of retrieved node IDs (in rank order).
        relevant_ids: Set of relevant node IDs for this query.
    
    Returns:
        MRR in range [0, 1]. Returns 0 if no relevant item found.
    """
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k.
    
    Args:
        retrieved_ids: List of retrieved node IDs (in rank order).
        relevant_ids: Set of relevant node IDs for this query.
        k: Cutoff rank.
    
    Returns:
        Recall@k in range [0, 1].
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        retrieved_ids: List of retrieved node IDs (in rank order).
        relevant_ids: Set of relevant node IDs for this query.
        k: Cutoff rank.
    
    Returns:
        NDCG@k in range [0, 1].
    """
    top_k = retrieved_ids[:k]
    
    # DCG: sum of (relevance / log(rank+1))
    dcg = sum(
        1.0 / np.log2(rank + 2)  # log2(rank+1) but rank is 0-indexed
        for rank, rid in enumerate(top_k)
        if rid in relevant_ids
    )
    
    # IDCG: ideal DCG with all relevant items first
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_k))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


# =========================================================================
# Evaluation Runner
# =========================================================================


class EvaluationRunner:
    """Orchestrates retrieval evaluation against golden queries."""

    def __init__(self, pipeline: RAGPipeline, golden_queries: dict[str, Any]):
        self.pipeline = pipeline
        self.golden_queries = golden_queries
        self.results: list[dict[str, Any]] = []

    async def run(self, top_k: int = 10) -> dict[str, Any]:
        """Run evaluation on all golden queries.
        
        Args:
            top_k: Number of results to retrieve and evaluate.
        
        Returns:
            Dictionary with aggregated metrics and per-query results.
        """
        queries = self.golden_queries.get("queries", [])
        print(f"\n[*] Running evaluation on {len(queries)} queries (top_k={top_k})...\n")

        for query_data in queries:
            result = await self._evaluate_query(query_data, top_k)
            self.results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_metrics(top_k)
        return aggregated

    async def _evaluate_query(
        self,
        query_data: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        """Evaluate a single query.
        
        Args:
            query_data: Query object from golden set.
            top_k: Number of results to retrieve.
        
        Returns:
            Result dict with query ID, metrics, and retrieved chunks.
        """
        query_id = query_data.get("id", "unknown")
        query_text = query_data.get("query", "")
        expected_sections = set(query_data.get("expected_sections", []))

        print(f"  [{query_id}] {query_text}")

        try:
            t0 = time.time()
            search_result = await self.pipeline.search(query_text, top_k=top_k)
            latency_ms = time.time() - t0

            # Extract retrieved section paths as proxy for relevance
            retrieved_sections = [
                chunk.metadata.get("section_path", "")
                for chunk in search_result.chunks
            ]

            # Build a set of matched expected sections (fuzzy match)
            matched_sections = set()
            for ret_sec in retrieved_sections:
                for exp_sec in expected_sections:
                    if exp_sec.lower() in ret_sec.lower() or ret_sec.lower() in exp_sec.lower():
                        matched_sections.add(exp_sec)

            # Compute metrics
            prec = precision_at_k(retrieved_sections, expected_sections, top_k)
            mrr = mean_reciprocal_rank(retrieved_sections, expected_sections)
            recall = recall_at_k(retrieved_sections, expected_sections, top_k)
            ndcg = ndcg_at_k(retrieved_sections, expected_sections, top_k)

            result = {
                "query_id": query_id,
                "query": query_text,
                "difficulty": query_data.get("difficulty", "unknown"),
                "type": query_data.get("type", "unknown"),
                "expected_sections": list(expected_sections),
                "matched_sections": list(matched_sections),
                "retrieved_sections": retrieved_sections,
                "num_expected": len(expected_sections),
                "num_matched": len(matched_sections),
                "precision_at_k": round(prec, 4),
                "mrr": round(mrr, 4),
                "recall_at_k": round(recall, 4),
                "ndcg_at_k": round(ndcg, 4),
                "latency_ms": round(latency_ms * 1000, 2),
            }

            print(f"      Precision@{top_k}: {prec:.4f} | MRR: {mrr:.4f} | "
                  f"Recall@{top_k}: {recall:.4f} | NDCG@{top_k}: {ndcg:.4f}")

        except Exception as e:
            print(f"      ERROR: {e}")
            result = {
                "query_id": query_id,
                "query": query_text,
                "error": str(e),
            }

        return result

    def _aggregate_metrics(self, top_k: int) -> dict[str, Any]:
        """Compute aggregate metrics across all queries.
        
        Args:
            top_k: The k value used for retrieval.
        
        Returns:
            Dictionary with aggregate metrics.
        """
        # Filter out error results
        valid_results = [r for r in self.results if "error" not in r]

        if not valid_results:
            return {"error": "No valid results to aggregate"}

        # Compute means
        precisions = [r["precision_at_k"] for r in valid_results]
        mrrs = [r["mrr"] for r in valid_results]
        recalls = [r["recall_at_k"] for r in valid_results]
        ndcgs = [r["ndcg_at_k"] for r in valid_results]
        latencies = [r["latency_ms"] for r in valid_results]

        aggregated = {
            "summary": {
                "total_queries": len(self.results),
                "valid_queries": len(valid_results),
                "top_k": top_k,
            },
            "aggregate_metrics": {
                f"precision_at_{top_k}": {
                    "mean": round(np.mean(precisions), 4),
                    "std": round(np.std(precisions), 4),
                    "min": round(np.min(precisions), 4),
                    "max": round(np.max(precisions), 4),
                },
                "mrr": {
                    "mean": round(np.mean(mrrs), 4),
                    "std": round(np.std(mrrs), 4),
                    "min": round(np.min(mrrs), 4),
                    "max": round(np.max(mrrs), 4),
                },
                f"recall_at_{top_k}": {
                    "mean": round(np.mean(recalls), 4),
                    "std": round(np.std(recalls), 4),
                    "min": round(np.min(recalls), 4),
                    "max": round(np.max(recalls), 4),
                },
                f"ndcg_at_{top_k}": {
                    "mean": round(np.mean(ndcgs), 4),
                    "std": round(np.std(ndcgs), 4),
                    "min": round(np.min(ndcgs), 4),
                    "max": round(np.max(ndcgs), 4),
                },
                "latency_ms": {
                    "mean": round(np.mean(latencies), 2),
                    "std": round(np.std(latencies), 2),
                    "min": round(np.min(latencies), 2),
                    "max": round(np.max(latencies), 2),
                },
            },
            "per_query_results": self.results,
        }

        return aggregated

    def print_summary(self, aggregated: dict[str, Any]) -> None:
        """Pretty-print evaluation summary.
        
        Args:
            aggregated: Aggregated metrics dictionary.
        """
        if "error" in aggregated:
            print(f"\n[!] {aggregated['error']}")
            return

        summary = aggregated["summary"]
        metrics = aggregated["aggregate_metrics"]

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal Queries: {summary['total_queries']} | "
              f"Valid: {summary['valid_queries']} | top_k: {summary['top_k']}")

        print("\n--- Aggregate Metrics ---\n")

        for metric_name, values in metrics.items():
            if isinstance(values, dict) and "mean" in values:
                mean = values["mean"]
                std = values["std"]
                min_v = values["min"]
                max_v = values["max"]
                print(f"{metric_name:25s}  Mean: {mean:.4f}  ±{std:.4f}  "
                      f"[{min_v:.4f} - {max_v:.4f}]")

        print("\n" + "=" * 70 + "\n")


# =========================================================================
# Main
# =========================================================================


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline retrieval quality."
    )
    parser.add_argument(
        "--golden-set",
        type=str,
        default="evaluation/golden_queries.json",
        help="Path to golden query set JSON file.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default="test-doc",
        help="Document ID to use for ingest (optional; uses existing if available).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve for evaluation (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results.json",
        help="Output file for detailed results.",
    )

    args = parser.parse_args()

    # Load golden queries
    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"[!] Golden query set not found: {golden_path}")
        return

    with open(golden_path) as f:
        golden_queries = json.load(f)

    # Initialize pipeline
    print("[*] Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    pipeline._ensure_loaded_from_store()

    if pipeline.chunk_index is None:
        print("[!] No documents ingested yet. Pipeline is empty.")
        print(f"[*] Tip: Use 'python -m cli ingest --file <path> --doc-id {args.doc_id}' first.")
        return

    # Run evaluation
    runner = EvaluationRunner(pipeline, golden_queries)
    aggregated = await runner.run(top_k=args.top_k)

    # Print summary
    runner.print_summary(aggregated)

    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[*] Detailed results saved to: {output_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
