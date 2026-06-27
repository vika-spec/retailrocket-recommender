"""
evaluator.py – Full GRU4Rec evaluation (academic + business + production metrics).
Now includes MMR (Maximum Marginal Relevance) re-ranking to fix category diversity.
"""

import numpy as np
import torch
import logging
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from config import MODEL_DIR, TEST_SESSIONS, TRAIN_SESSIONS, TRAIN_ITEM_FEATURES
# Also add these if not already present:
from config import ITEM_PROPERTIES_1, ITEM_PROPERTIES_2

from business_metric import (
    compute_user_metrics,
    evaluate_cold_start,
    compute_novelty,
    compute_information_content,
    compute_avg_diversity,
    get_model_size_mb,
    get_cpu_ram_usage_mb,
    compute_unknown_ratio,
    evaluate_by_session_length,
    evaluate_by_user_activity,
    compute_stability,
    compute_category_diversity,
    mmr_rerank,  # <-- NEW IMPORT
)
from metrics import recall_at_k, mrr_at_k, ndcg_at_k, hitrate_at_k
from dataloader import create_eval_dataloader
from app.models import BaseGRU

logger = logging.getLogger(__name__)


def evaluate_gru4rec(
    test_sessions: List[Dict[str, Any]],
    model: torch.nn.Module,
    item_encoder: Dict[int, int],
    device: torch.device,
    k_values: List[int],
    max_seq_len: int,
    gap_max: float,
    batch_size: int = 2048,
    user_ids_list: Optional[List[int]] = None,
    item_counts: Optional[Dict[int, int]] = None,
    total_train_interactions: Optional[int] = None,
    popularity_buckets: Optional[Dict[str, set]] = None,
    user_activity_counts: Optional[Dict[int, int]] = None,
    item_to_category: Optional[Dict[int, int]] = None,
    # NEW PARAMETERS for MMR
    mmr_lambda: float = 0.5,              # 0.5 = balanced, 0.3 = more diversity
    mmr_pool_size_multiplier: int = 3,    # How many candidates to fetch before re-ranking
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of GRU4Rec with optional MMR re-ranking.
    """
    model.to(device)
    model.eval()

    # GPU memory tracking
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    use_amp = device.type == 'cuda'
    logger.info(f"Batch size: {batch_size}, AMP: {use_amp}")
    logger.info(f"MMR Lambda: {mmr_lambda}, Pool Multiplier: {mmr_pool_size_multiplier}")

    idx_to_item = {v: k for k, v in item_encoder.items()}

    # ---- Accumulators ----
    results = {k: {'recall': [], 'mrr': [], 'ndcg': [], 'hitrate': []} for k in k_values}
    user_predictions = defaultdict(list)
    group_predictions = {'head': [], 'medium': [], 'tail': []}
    recommended_lists = []
    batch_latencies = []
    total_prefixes = 0
    recommended_items_flat = []
    session_length_predictions = defaultdict(list)
    activity_predictions = defaultdict(list)
    user_consecutive_lists = defaultdict(list)

    head_set = popularity_buckets.get('head', set()) if popularity_buckets else set()
    medium_set = popularity_buckets.get('medium', set()) if popularity_buckets else set()
    tail_set = popularity_buckets.get('tail', set()) if popularity_buckets else set()

    # ---- Create DataLoader ----
    dataloader, filtered_sessions, all_test_items_original = create_eval_dataloader(
        test_sessions=test_sessions,
        item_encoder=item_encoder,
        max_seq_len=max_seq_len,
        gap_max=gap_max,
        batch_size=batch_size,
        user_ids_list=user_ids_list,
    )

    total_prefixes_data = len(dataloader.dataset)
    total_batches = (total_prefixes_data + batch_size - 1) // batch_size
    logger.info(f"Total prefixes: {total_prefixes_data:,} → {total_batches} batches")

    # ---- Main evaluation loop ----
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="GRU4Rec (MMR)", total=total_batches)):
        (
            items_batch, events_batch, hours_batch, days_batch, gaps_batch,
            targets_batch, target_orig_batch, hist_lens_batch, user_ids_batch
        ) = batch

        items_t = items_batch.to(device, non_blocking=True)
        events_t = events_batch.to(device, non_blocking=True)
        hour_t = hours_batch.to(device, non_blocking=True)
        day_t = days_batch.to(device, non_blocking=True)
        gap_t = gaps_batch.to(device, non_blocking=True)

        start_time = time.perf_counter()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            with torch.no_grad():
                scores = model(items_t, events_t, hour_t, day_t, gap_t)
        batch_latencies.append(time.perf_counter() - start_time)

        scores_np = scores.cpu().numpy()

        del items_t, events_t, hour_t, day_t, gap_t, scores
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        max_k = max(k_values)
        
        # --- MMR Pool Size ---
        pool_size = max(50, max_k * mmr_pool_size_multiplier)
        # Ensure we don't request more items than the vocabulary size
        if pool_size >= scores_np.shape[1]:
            pool_size = scores_np.shape[1] - 1
        if pool_size < max_k:
            pool_size = max_k  # fallback

        # Process each item in the batch
        for i in range(len(targets_batch)):
            target = targets_batch[i].item()
            target_orig = target_orig_batch[i].item()
            hist_len = hist_lens_batch[i].item()
            user_id = user_ids_batch[i].item()

            # ---- 1. Get top candidates (pool) ----
            top_indices = np.argpartition(-scores_np[i], pool_size)[:pool_size]
            top_indices = top_indices[np.argsort(-scores_np[i][top_indices])]

            candidate_items = []
            candidate_scores = []
            for idx in top_indices:
                if idx in idx_to_item and idx != 0:
                    candidate_items.append(int(idx))
                    candidate_scores.append(float(scores_np[i][idx]))
                    if len(candidate_items) >= pool_size:
                        break

            # ---- 2. Apply MMR re-ranking (if we have category mapping) ----
            if item_to_category and len(candidate_items) > max_k:
                pred_encoded = mmr_rerank(
                    candidate_indices=candidate_items,
                    candidate_scores=candidate_scores,
                    item_to_category=item_to_category,
                    top_k=max_k,
                    lambda_=mmr_lambda
                )
            else:
                # Fallback: just take top max_k
                pred_encoded = candidate_items[:max_k]

            # ---- 3. Compute metrics for all k_values ----
            for k in k_values:
                pred_k = pred_encoded[:k]
                results[k]['recall'].append(recall_at_k(pred_k, target, k))
                results[k]['mrr'].append(mrr_at_k(pred_k, target, k))
                results[k]['ndcg'].append(ndcg_at_k(pred_k, target, k))
                results[k]['hitrate'].append(hitrate_at_k(pred_k, target, k))

            # ---- 4. Populate all other accumulators (same as before) ----
            user_predictions[user_id].append({'target': target, 'preds': pred_encoded})

            if target_orig in head_set:
                group_predictions['head'].append({'target': target, 'preds': pred_encoded})
            elif target_orig in medium_set:
                group_predictions['medium'].append({'target': target, 'preds': pred_encoded})
            elif target_orig in tail_set:
                group_predictions['tail'].append({'target': target, 'preds': pred_encoded})

            recommended_lists.append(pred_encoded[:20])
            recommended_items_flat.extend(pred_encoded[:20])

            # Session length grouping
            if hist_len < 5:
                length_group = '<5'
            elif hist_len < 10:
                length_group = '5-10'
            elif hist_len < 20:
                length_group = '10-20'
            else:
                length_group = '20+'
            session_length_predictions[length_group].append(
                {'target': target, 'preds': pred_encoded}
            )

            if user_activity_counts is not None:
                user_act = user_activity_counts.get(user_id, 0)
                if user_act < 5:
                    act_group = 'new'
                elif user_act < 20:
                    act_group = 'medium'
                else:
                    act_group = 'heavy'
                activity_predictions[act_group].append(
                    {'target': target, 'preds': pred_encoded}
                )

            user_consecutive_lists[user_id].append(pred_encoded[:20])
            total_prefixes += 1

    # ---- Aggregate metrics (NO CHANGES NEEDED HERE) ----
    metrics = {}

    for k in k_values:
        metrics[f'recall@{k}'] = np.mean(results[k]['recall']) or 0.0
        metrics[f'mrr@{k}'] = np.mean(results[k]['mrr']) or 0.0
        metrics[f'ndcg@{k}'] = np.mean(results[k]['ndcg']) or 0.0
        metrics[f'hitrate@{k}'] = np.mean(results[k]['hitrate']) or 0.0

    metrics['coverage@20'] = (len(set(recommended_items_flat)) / len(item_encoder)
                              if item_encoder else 0.0)
    metrics['total_prefixes'] = total_prefixes

    user_metrics = compute_user_metrics(user_predictions, k_values)
    metrics['user_recall@20'] = user_metrics.get('recall@20', 0.0)
    metrics['user_mrr@20'] = user_metrics.get('mrr@20', 0.0)
    metrics['user_ndcg@20'] = user_metrics.get('ndcg@20', 0.0)

    cold_metrics = evaluate_cold_start(group_predictions, k_values)
    for group in ('head', 'medium', 'tail'):
        metrics[f'cold_{group}_recall@20'] = cold_metrics.get(group, {}).get('recall@20', 0.0)
        metrics[f'cold_{group}_mrr@20'] = cold_metrics.get(group, {}).get('mrr@20', 0.0)

    if item_counts is not None and total_train_interactions is not None:
        metrics['novelty_info_content'] = compute_novelty(
            recommended_lists, item_counts, total_train_interactions
        )
        metrics['information_content'] = compute_information_content(
            recommended_lists, item_counts, total_train_interactions
        )
    else:
        metrics['novelty_info_content'] = 0.0
        metrics['information_content'] = 0.0

    try:
        item_embeddings = model.item_emb.weight.detach().cpu().numpy()
        metrics['diversity_intra_list'] = compute_avg_diversity(recommended_lists, item_embeddings)
    except AttributeError:
        logger.warning("Model has no 'item_emb' attribute. Skipping diversity.")
        metrics['diversity_intra_list'] = 0.0

    if batch_latencies:
        metrics['latency_avg_ms'] = np.mean(batch_latencies) * 1000
        metrics['latency_p95_ms'] = np.percentile(batch_latencies, 95) * 1000
        metrics['latency_p99_ms'] = np.percentile(batch_latencies, 99) * 1000
    else:
        metrics['latency_avg_ms'] = 0.0
        metrics['latency_p95_ms'] = 0.0
        metrics['latency_p99_ms'] = 0.0

    metrics['model_size_mb'] = get_model_size_mb(model)
    metrics['peak_gpu_memory_mb'] = (torch.cuda.max_memory_allocated(device) / 1024**2
                                     if device.type == 'cuda' else 0.0)
    metrics['cpu_ram_mb'] = get_cpu_ram_usage_mb()
    metrics['unknown_item_ratio'] = compute_unknown_ratio(all_test_items_original, item_encoder)

    session_metrics = evaluate_by_session_length(session_length_predictions, k_values)
    for group in ('<5', '5-10', '10-20', '20+'):
        metrics[f'session_recall@20_{group}'] = session_metrics.get(f'recall@20_{group}', 0.0)

    if user_activity_counts is not None:
        activity_metrics = evaluate_by_user_activity(activity_predictions, k_values)
        for group in ('new', 'medium', 'heavy'):
            metrics[f'activity_recall@20_{group}'] = activity_metrics.get(f'recall@20_{group}', 0.0)
    else:
        for group in ('new', 'medium', 'heavy'):
            metrics[f'activity_recall@20_{group}'] = 0.0

    stabilities = [compute_stability(lst) for lst in user_consecutive_lists.values() if len(lst) > 1]
    metrics['stability_jaccard'] = np.mean(stabilities) if stabilities else 1.0

    if item_to_category is not None and recommended_lists:
        original_lists = [
            [idx_to_item.get(item) for item in lst if item in idx_to_item]
            for lst in recommended_lists
        ]
        metrics['category_diversity'] = compute_category_diversity(original_lists, item_to_category)
    else:
        metrics['category_diversity'] = 0.0

    return metrics


# ======================================================================
# Helper functions (SAME AS BEFORE — NO CHANGES NEEDED)
# ======================================================================

def load_test_sessions() -> pd.DataFrame:
    if not TEST_SESSIONS.exists():
        raise FileNotFoundError(
            f"Test sessions not found at {TEST_SESSIONS}\n"
            "Run preprocessing.py first with temporal split."
        )
    logger.info(f"Loading test sessions from: {TEST_SESSIONS}")
    return pd.read_parquet(TEST_SESSIONS)


def load_model_and_metadata(model_dir: Path):
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    with open(model_dir / 'item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    with open(model_dir / 'item_ids.pkl', 'rb') as f:
        item_ids = pickle.load(f)
    with open(model_dir / 'gap_max.pkl', 'rb') as f:
        gap_max = pickle.load(f)

    num_items = config.get('num_items', len(item_ids))
    num_events = config.get('num_events', 4)

    model = BaseGRU(
        num_items=num_items,
        num_events=num_events,
        embedding_dim=config.get('embedding_dim', 64),
        event_embedding_dim=config.get('event_embedding_dim', 8),
        hidden_size=config.get('hidden_size', 224),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
    )
    state_dict = torch.load(model_dir / 'model_best.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully.")
    return model, config, item_encoder, item_ids, gap_max


def get_item_counts_from_train() -> Dict[int, int]:
    if not TRAIN_SESSIONS.exists():
        logger.warning("Training sessions not found. Using empty counts.")
        return {}
    train_df = pd.read_parquet(TRAIN_SESSIONS)
    all_items = []
    for seq in train_df['item_sequence']:
        all_items.extend(seq)
    return dict(pd.Series(all_items).value_counts().to_dict())


def get_total_train_interactions(train_df: pd.DataFrame) -> int:
    if train_df is None:
        return 0
    return sum(len(seq) for seq in train_df['item_sequence'])


def get_user_activity_counts(train_df: pd.DataFrame) -> Dict[int, int]:
    if train_df is None:
        return {}
    counts = defaultdict(int)
    for _, row in train_df.iterrows():
        counts[row['visitorid']] += len(row['item_sequence'])
    return dict(counts)


def load_category_mapping() -> Optional[Dict[int, int]]:
    """
    Load category mapping from raw item_properties files.
    The category is stored as property == 'categoryid'.
    """
    # Try loading from processed features first (fastest)
    if TRAIN_ITEM_FEATURES.exists():
        try:
            df = pd.read_parquet(TRAIN_ITEM_FEATURES)
            if 'itemid' in df.columns and 'categoryid' in df.columns:
                # Only keep items with valid categories (not -1)
                df = df[df['categoryid'] != -1]
                mapping = dict(zip(df['itemid'].astype(int), df['categoryid'].astype(int)))
                logger.info(f"✅ Loaded category mapping for {len(mapping):,} items from processed features.")
                return mapping
        except Exception as e:
            logger.warning(f"Could not load from processed features: {e}")
    
    # Fallback: Build from raw properties files
    logger.info("Building category mapping from raw item_properties files...")
    
    if not ITEM_PROPERTIES_1.exists() or not ITEM_PROPERTIES_2.exists():
        logger.warning("Item properties files not found. MMR will be disabled.")
        return None
    
    try:
        # Load both parts
        props1 = pd.read_csv(ITEM_PROPERTIES_1, dtype={"itemid": "int32"})
        props2 = pd.read_csv(ITEM_PROPERTIES_2, dtype={"itemid": "int32"})
        props = pd.concat([props1, props2], ignore_index=True)
        
        # Clean column names
        props.columns = [c.strip().lower() for c in props.columns]
        
        # Filter only categoryid properties
        cat_props = props[props['property'] == 'categoryid'][['itemid', 'value']].copy()
        
        # Convert category values to integers
        cat_props['categoryid'] = pd.to_numeric(cat_props['value'], errors='coerce')
        cat_props = cat_props.dropna(subset=['categoryid'])
        cat_props['categoryid'] = cat_props['categoryid'].astype(int)
        
        # Remove duplicates (keep first occurrence)
        cat_props = cat_props.drop_duplicates('itemid')
        
        # Create mapping
        mapping = dict(zip(cat_props['itemid'].astype(int), cat_props['categoryid'].astype(int)))
        
        logger.info(f"✅ Loaded category mapping for {len(mapping):,} items from raw properties.")
        
        # Check coverage
        if len(mapping) == 0:
            logger.warning("No categories found in properties files.")
            return None
            
        return mapping
        
    except Exception as e:
        logger.error(f"Failed to build category mapping: {e}")
        return None

def print_results_table(gru4rec_metrics: Dict, pop_metrics: Dict, k_values: List[int]):
    print(f"\n{'='*80}")
    print("FULL EVALUATION RESULTS (Academic + Business + Production Metrics)")
    print('='*80)
    print(f"\n{'Metric':<30} {'GRU4Rec':<12} {'Popularity':<12} {'Improvement':<12}")
    print('-'*75)

    for k in k_values:
        gru = gru4rec_metrics[f'recall@{k}']
        pop = pop_metrics[f'recall@{k}']
        impr = (gru / pop) if pop > 0 else float('inf')
        print(f"Recall@{k:<2}                 {gru:.4f}      {pop:.4f}      {impr:.1f}x")
    for k in k_values:
        gru = gru4rec_metrics[f'mrr@{k}']
        pop = pop_metrics[f'mrr@{k}']
        impr = (gru / pop) if pop > 0 else float('inf')
        print(f"MRR@{k:<2}                   {gru:.4f}      {pop:.4f}      {impr:.1f}x")
    for k in k_values:
        gru = gru4rec_metrics[f'ndcg@{k}']
        pop = pop_metrics[f'ndcg@{k}']
        impr = (gru / pop) if pop > 0 else float('inf')
        print(f"NDCG@{k:<2}                  {gru:.4f}      {pop:.4f}      {impr:.1f}x")
    for k in k_values:
        gru = gru4rec_metrics[f'hitrate@{k}']
        pop = pop_metrics[f'hitrate@{k}']
        impr = (gru / pop) if pop > 0 else float('inf')
        print(f"HitRate@{k:<2}               {gru:.4f}      {pop:.4f}      {impr:.1f}x")
    print('-'*75)
    print(f"{'Coverage@20':<30} {gru4rec_metrics['coverage@20']:.4f}      {pop_metrics.get('coverage@20', 0.0):.4f}")
    print(f"{'Total Prefixes':<30} {gru4rec_metrics['total_prefixes']:,}")
    print()

    print(f"{'User Recall@20':<30} {gru4rec_metrics.get('user_recall@20', 0.0):.4f}")
    print(f"{'User MRR@20':<30} {gru4rec_metrics.get('user_mrr@20', 0.0):.4f}")
    print()

    print(f"{'Cold Head Recall@20':<30} {gru4rec_metrics.get('cold_head_recall@20', 0.0):.4f}")
    print(f"{'Cold Medium Recall@20':<30} {gru4rec_metrics.get('cold_medium_recall@20', 0.0):.4f}")
    print(f"{'Cold Tail Recall@20':<30} {gru4rec_metrics.get('cold_tail_recall@20', 0.0):.4f}")
    print()

    print(f"{'Novelty (Info Content)':<30} {gru4rec_metrics.get('novelty_info_content', 0.0):.4f}")
    print(f"{'Information Content':<30} {gru4rec_metrics.get('information_content', 0.0):.4f}")
    print(f"{'Diversity (Intra-list)':<30} {gru4rec_metrics.get('diversity_intra_list', 0.0):.4f}")
    print()

    print(f"{'Latency Avg (ms)':<30} {gru4rec_metrics.get('latency_avg_ms', 0.0):.2f}")
    print(f"{'Latency P95 (ms)':<30} {gru4rec_metrics.get('latency_p95_ms', 0.0):.2f}")
    print(f"{'Latency P99 (ms)':<30} {gru4rec_metrics.get('latency_p99_ms', 0.0):.2f}")
    print(f"{'Model Size (MB)':<30} {gru4rec_metrics.get('model_size_mb', 0.0):.2f}")
    print(f"{'Peak GPU Memory (MB)':<30} {gru4rec_metrics.get('peak_gpu_memory_mb', 0.0):.2f}")
    print(f"{'CPU RAM (MB)':<30} {gru4rec_metrics.get('cpu_ram_mb', 0.0):.2f}")
    print()

    print(f"{'Unknown Item Ratio':<30} {gru4rec_metrics.get('unknown_item_ratio', 0.0):.4f}")
    print(f"{'Stability (Jaccard)':<30} {gru4rec_metrics.get('stability_jaccard', 0.0):.4f}")
    print(f"{'Category Diversity':<30} {gru4rec_metrics.get('category_diversity', 0.0):.4f}")  # <-- THIS WILL NOW BE >0!
    print()

    print("--- Session Length Recall@20 ---")
    for group in ('<5', '5-10', '10-20', '20+'):
        print(f"  {group:>5} : {gru4rec_metrics.get(f'session_recall@20_{group}', 0.0):.4f}")
    print()

    print("--- User Activity Recall@20 ---")
    for group in ('new', 'medium', 'heavy'):
        print(f"  {group:>6} : {gru4rec_metrics.get(f'activity_recall@20_{group}', 0.0):.4f}")
    print('='*80)