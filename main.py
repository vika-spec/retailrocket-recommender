"""
main.py – Entry point for full evaluation.
Loads data, runs GRU4Rec and popularity baseline, prints and saves results.
"""
"""
main.py – Entry point for full evaluation.
Loads data, runs GRU4Rec and popularity baseline, prints and saves results.
"""

import sys
import time
import json
import logging
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, TEST_SESSIONS, TRAIN_SESSIONS
from business_metric import get_popularity_buckets
from metrics import PopularityBaseline
from evaluator import (
    evaluate_gru4rec,
    load_test_sessions,
    load_model_and_metadata,
    get_item_counts_from_train,
    get_total_train_interactions,
    get_user_activity_counts,
    load_category_mapping,
    print_results_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def main() -> None:
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("FULL EVALUATION: GRU4Rec + ALL PRODUCTION METRICS")
    logger.info("=" * 80)

    # 1. Load test sessions
    test_df = load_test_sessions()
    logger.info(f"Test sessions: {len(test_df):,}")

    visitor_ids = test_df['visitorid'].tolist()
    test_sessions = [
        {
            'visitorid': row['visitorid'],
            'item_seq': list(row['item_sequence']),
            'event_seq': list(row.get('event_sequence', [])),
            'hour_seq': list(row.get('hour_sequence', [])),
            'day_seq': list(row.get('day_sequence', [])),
            'time_gap_seq': list(row.get('time_gap_sequence', [])),
        }
        for _, row in test_df.iterrows()
    ]
    total_prefixes_approx = sum(len(s['item_seq']) - 1 for s in test_sessions)
    logger.info(f"Approx test prefixes: {total_prefixes_approx:,}")

    # 2. Load model
    model, config, item_encoder, _, gap_max = load_model_and_metadata(MODEL_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Item encoder size: {len(item_encoder):,}")

    k_values = [5, 10, 20]
    batch_size = 2048

    # 3. Load training data
    train_df = None
    if TRAIN_SESSIONS.exists():
        train_df = pd.read_parquet(TRAIN_SESSIONS)
        logger.info(f"Training sessions: {len(train_df):,}")
    else:
        logger.warning("Training sessions not found. Some metrics will be missing.")

    item_counts = get_item_counts_from_train()
    total_train_interactions = get_total_train_interactions(train_df) if train_df is not None else 0
    user_activity_counts = get_user_activity_counts(train_df) if train_df is not None else {}

    # 4. Popularity buckets
    popularity_buckets = get_popularity_buckets(
        item_counts, method='percentile',
        head_ratio=0.1, medium_ratio=0.4, tail_ratio=0.5
    )
    logger.info(
        f"Cold‑start buckets: head={len(popularity_buckets['head'])}, "
        f"medium={len(popularity_buckets['medium'])}, tail={len(popularity_buckets['tail'])}"
    )

    # 5. Category mapping
    item_to_category = load_category_mapping()

    # 6. Evaluate GRU4Rec
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING GRU4Rec (full production metrics)")
    logger.info("=" * 80)

    gru4rec_metrics = evaluate_gru4rec(
    test_sessions=test_sessions,
    model=model,
    item_encoder=item_encoder,
    device=device,
    k_values=k_values,
    max_seq_len=config['max_seq_len'],
    gap_max=gap_max,
    batch_size=batch_size,
    user_ids_list=visitor_ids,
    item_counts=item_counts,
    total_train_interactions=total_train_interactions,
    popularity_buckets=popularity_buckets,
    user_activity_counts=user_activity_counts,
    item_to_category=item_to_category,
    mmr_lambda=0.5,          # <-- NEW: tune this (0.3 to 0.7)
    mmr_pool_size_multiplier=3,  # <-- NEW: fetch 3x candidates
)

    # 7. Evaluate Popularity Baseline
    pop_baseline = PopularityBaseline(item_counts, top_k=max(k_values))
    pop_metrics = pop_baseline.evaluate(test_sessions, item_encoder, k_values)

    # 8. Print results
    print_results_table(gru4rec_metrics, pop_metrics, k_values)

    # 9. Save results
    end_time = time.time()
    eval_time = end_time - start_time
    total_prefixes = gru4rec_metrics['total_prefixes']
    throughput = total_prefixes / eval_time if eval_time > 0 else 0

    logger.info(f"\nEvaluation time: {eval_time:.2f}s, Throughput: {throughput:.1f} prefixes/sec")

    results = {
        'dataset': 'RetailRocket',
        'model': 'GRU4Rec',
        'k_values': k_values,
        'gru4rec_metrics': gru4rec_metrics,
        'popularity_metrics': pop_metrics,
        'metadata': {
            'test_sessions': len(test_sessions),
            'num_items': len(item_encoder),
            'batch_size': batch_size,
            'max_seq_len': config['max_seq_len'],
            'eval_time_seconds': eval_time,
            'throughput_prefixes_per_sec': throughput,
        }
    }

    # Convert numpy types to Python native types for JSON serialization
    results_serializable = convert_to_serializable(results)

    output_path = Path(__file__).parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()