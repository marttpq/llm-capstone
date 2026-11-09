import os
import json
import time
import sys
import types

# Windows WDAC / AppLocker fallback for jiter binary DLL
try:
    import jiter  # noqa: F401
except ImportError:
    jiter_shim = types.ModuleType("jiter")
    jiter_shim.from_json = lambda b, **kwargs: json.loads(
        b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b
    )
    jiter_shim.cache_clear = lambda: None
    sys.modules["jiter"] = jiter_shim

from dotenv import load_dotenv
from openai import OpenAI
from rag import init_knowledge_base, vector_search

# Load environment configuration
load_dotenv()


def load_eval_dataset(filepath: str = "eval_set.json"):
    """Loads the evaluation dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evaluation dataset '{filepath}' not found.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(dataset, client: OpenAI, top_k: int):
    """
    Runs the retrieval evaluation across all test cases for a given top_k.
    Computes Hit Rate / Recall@K by checking if expected_keyword is present
    in the retrieved chunk text.
    """
    total = len(dataset)
    hits = 0
    failures = []
    latencies = []

    for item in dataset:
        question = item["question"]
        expected_keyword = item["expected_keyword"]

        start_time = time.perf_counter()
        retrieved_results = vector_search(query=question, client=client, top_k=top_k)
        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)

        # Concatenate retrieved chunk contents
        combined_text = " ".join([r["content"] for r in retrieved_results])

        # Substring keyword check (case-insensitive)
        is_hit = expected_keyword.lower() in combined_text.lower()

        if is_hit:
            hits += 1
        else:
            failures.append({
                "question": question,
                "expected_keyword": expected_keyword,
                "retrieved_citations": [r["citation"] for r in retrieved_results],
            })

    hit_rate = (hits / total) * 100 if total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "top_k": top_k,
        "total": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "avg_latency_ms": avg_latency,
        "failures": failures,
    }


def main():
    print("=" * 80)
    print("          RAG RETRIEVAL EVALUATION HARNESS (QA AUTOMATION)")
    print("=" * 80)

    # 1. Initialize OpenAI Client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    # 2. Index Knowledge Base
    kb_path = "knowledge.txt"
    print(f"[*] Pre-indexing knowledge base from '{kb_path}' into memory...")
    init_knowledge_base(kb_path, client)
    print("[+] Knowledge base indexing complete.\n")

    # 3. Load Test Cases
    dataset = load_eval_dataset("eval_set.json")
    print(f"[*] Loaded {len(dataset)} test cases from eval_set.json.\n")

    # 4. Run Baseline (top_k=1)
    print(f"[*] Executing Run 1: Baseline (top_k=1)...")
    baseline = evaluate_retrieval(dataset, client, top_k=1)

    # 5. Run Improved (top_k=3)
    print(f"[*] Executing Run 2: Improved (top_k=3)...")
    improved = evaluate_retrieval(dataset, client, top_k=3)

    # 6. Compute Differences & Delta
    diff_hits = improved["hits"] - baseline["hits"]
    diff_abs_pct = improved["hit_rate"] - baseline["hit_rate"]
    diff_rel_pct = (
        (diff_hits / baseline["hits"]) * 100 if baseline["hits"] > 0 else 0.0
    )

    # 7. Print Summary Report Table
    print("\n" + "=" * 80)
    print("                     EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Configuration':<22} | {'Hits / Total':<14} | {'Hit Rate (%)':<14} | {'Avg Latency':<12}"
    )
    print("-" * 80)
    print(
        f"{'Baseline (top_k=1)':<22} | {baseline['hits']:>2}/{baseline['total']} ({baseline['hit_rate']:5.1f}%) | "
        f"{baseline['hit_rate']:>12.2f}% | {baseline['avg_latency_ms']:>8.1f} ms"
    )
    print(
        f"{'Improved (top_k=3)':<22} | {improved['hits']:>2}/{improved['total']} ({improved['hit_rate']:5.1f}%) | "
        f"{improved['hit_rate']:>12.2f}% | {improved['avg_latency_ms']:>8.1f} ms"
    )
    print("-" * 80)
    print(
        f"Delta Improvement      : +{diff_hits} hits (+{diff_abs_pct:.2f}% absolute | +{diff_rel_pct:.2f}% relative)"
    )
    print("=" * 80)

    # 8. Detailed Failure Analysis (Baseline Misses that Improved Recovered)
    recovered = [
        f for f in baseline["failures"]
        if f["question"] not in [m["question"] for m in improved["failures"]]
    ]

    if recovered:
        print("\n[+] Sample Cases Recovered by top_k=3 Expansion:")
        for idx, item in enumerate(recovered[:5], 1):
            print(f"  {idx}. Q: \"{item['question']}\"")
            print(f"     Expected Keyword: '{item['expected_keyword']}'")
            print(f"     Baseline Miss: Retrieved only {item['retrieved_citations']}\n")

    if improved["failures"]:
        print("\n[-] Remaining Unmatched Cases in top_k=3:")
        for idx, item in enumerate(improved["failures"], 1):
            print(f"  {idx}. Q: \"{item['question']}\"")
            print(f"     Expected Keyword: '{item['expected_keyword']}'\n")
    else:
        print("\n[✓] All 30 test cases successfully retrieved in Improved (top_k=3)!")

    print("=" * 80)


if __name__ == "__main__":
    main()

