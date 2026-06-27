"""
business_metrics.py
===================
Production‑grade offline metrics for recommendation systems.
Includes standard ranking, user‑level, cold‑start, novelty (information content),
diversity, latency, memory, session/user segmentation, stability, and business diversity.
"""

import numpy as np
import torch
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import psutil

# --------------------------------------------------------------
# 1. User‑level metrics (corrected loop)
# --------------------------------------------------------------
def compute_user_metrics(user_predictions: Dict[int, List[Dict]], k_values: List[int]) -> Dict:
    """
    user_predictions: {user_id: [{'target': int, 'preds': list}, ...]}
    Returns per‑user averaged metrics (recall, mrr, ndcg) for each k.
    """
    user_records = defaultdict(lambda: {k: {'recall': [], 'mrr': [], 'ndcg': []} for k in k_values})

    for uid, preds in user_predictions.items():
        for p in preds:
            target = p['target']
            pred_list = p['preds']
            for k in k_values:
                pred_k = pred_list[:k]
                rec = 1.0 if target in pred_k else 0.0
                mrr = 0.0
                ndcg = 0.0
                for i, item in enumerate(pred_k):
                    if item == target:
                        mrr = 1.0 / (i + 1)
                        ndcg = 1.0 / np.log2(i + 2)   # IDCG = 1 for rank 1
                        break
                user_records[uid][k]['recall'].append(rec)
                user_records[uid][k]['mrr'].append(mrr)
                user_records[uid][k]['ndcg'].append(ndcg)

    final = {}
    for k in k_values:
        recall_vals, mrr_vals, ndcg_vals = [], [], []
        for metrics in user_records.values():
            if metrics[k]['recall']:
                recall_vals.append(np.mean(metrics[k]['recall']))
                mrr_vals.append(np.mean(metrics[k]['mrr']))
                ndcg_vals.append(np.mean(metrics[k]['ndcg']))
        final[f'recall@{k}'] = np.mean(recall_vals) if recall_vals else 0.0
        final[f'mrr@{k}'] = np.mean(mrr_vals) if mrr_vals else 0.0
        final[f'ndcg@{k}'] = np.mean(ndcg_vals) if ndcg_vals else 0.0
    return final

# --------------------------------------------------------------
# 2. Cold‑start splitting (fixed thresholds / percentiles)
# --------------------------------------------------------------
def get_popularity_buckets(
    item_counts: Dict[int, int],
    method: str = 'percentile',
    head_ratio: float = 0.1,
    medium_ratio: float = 0.4,
    tail_ratio: float = 0.5
) -> Dict[str, set]:
    """
    Returns sets of item IDs for head, medium, tail.
    method: 'percentile' or 'threshold'.
        - percentile: top head_ratio are head, next medium_ratio are medium, rest tail.
        - threshold: uses absolute counts (e.g. count>=100 → head, count>=20 → medium, else tail)
    """
    if not item_counts:
        return {'head': set(), 'medium': set(), 'tail': set()}

    items = list(item_counts.keys())
    counts = np.array([item_counts[it] for it in items])
    sorted_indices = np.argsort(counts)[::-1]   # descending
    total = len(items)

    if method == 'percentile':
        head_end = int(head_ratio * total)
        medium_end = int((head_ratio + medium_ratio) * total)
        head = set(items[i] for i in sorted_indices[:head_end])
        medium = set(items[i] for i in sorted_indices[head_end:medium_end])
        tail = set(items[i] for i in sorted_indices[medium_end:])
    else:  # threshold
        head = {it for it, cnt in item_counts.items() if cnt >= 100}
        medium = {it for it, cnt in item_counts.items() if 20 <= cnt < 100}
        tail = {it for it, cnt in item_counts.items() if cnt < 20}
    return {'head': head, 'medium': medium, 'tail': tail}

# --------------------------------------------------------------
# 2b. Cold‑start evaluation (compute metrics per bucket)
# --------------------------------------------------------------
def evaluate_cold_start(group_predictions: Dict[str, List[Dict]], k_values: List[int]) -> Dict:
    """
    group_predictions: {'head': [{'target': int, 'preds': list}, ...], ...}
    Returns metrics per group for each k.
    """
    results = {}
    for group, preds in group_predictions.items():
        group_metrics = {k: {'recall': [], 'mrr': [], 'ndcg': []} for k in k_values}
        for p in preds:
            target = p['target']
            pred_list = p['preds']
            for k in k_values:
                pred_k = pred_list[:k]
                rec = 1.0 if target in pred_k else 0.0
                mrr = 0.0
                ndcg = 0.0
                for i, item in enumerate(pred_k):
                    if item == target:
                        mrr = 1.0 / (i + 1)
                        ndcg = 1.0 / np.log2(i + 2)
                        break
                group_metrics[k]['recall'].append(rec)
                group_metrics[k]['mrr'].append(mrr)
                group_metrics[k]['ndcg'].append(ndcg)
        results[group] = {}
        for k in k_values:
            results[group][f'recall@{k}'] = np.mean(group_metrics[k]['recall']) if group_metrics[k]['recall'] else 0.0
            results[group][f'mrr@{k}'] = np.mean(group_metrics[k]['mrr']) if group_metrics[k]['mrr'] else 0.0
            results[group][f'ndcg@{k}'] = np.mean(group_metrics[k]['ndcg']) if group_metrics[k]['ndcg'] else 0.0
    return results

# --------------------------------------------------------------
# 3. Novelty (Information Content) – corrected
# --------------------------------------------------------------
def compute_novelty(recommended_lists: List[List[int]], 
                    item_popularity: Dict[int, int],
                    total_interactions: int) -> float:
    """
    Novelty = average -log2(popularity(i)) over recommended items.
    Higher = more novel.
    """
    if not recommended_lists or total_interactions == 0:
        return 0.0
    novelty_scores = []
    for lst in recommended_lists:
        for item in lst:
            freq = item_popularity.get(item, 0)
            if freq > 0:
                prob = freq / total_interactions
                novelty_scores.append(-np.log2(prob))
    return np.mean(novelty_scores) if novelty_scores else 0.0

# --------------------------------------------------------------
# 4. Information Content (self‑information) – renamed
# --------------------------------------------------------------
def compute_information_content(recommended_lists: List[List[int]],
                                item_popularity: Dict[int, int],
                                total_interactions: int) -> float:
    """
    Same as novelty, but averaged per list (self‑information).
    """
    if not recommended_lists or total_interactions == 0:
        return 0.0
    list_scores = []
    for lst in recommended_lists:
        entropy = 0.0
        for item in lst:
            freq = item_popularity.get(item, 0)
            if freq > 0:
                prob = freq / total_interactions
                entropy -= np.log2(prob)
        list_scores.append(entropy / len(lst) if lst else 0.0)
    return np.mean(list_scores) if list_scores else 0.0

# --------------------------------------------------------------
# 5. Diversity (no scipy)
# --------------------------------------------------------------
def compute_intra_list_diversity(pred_list: List[int], item_embeddings: np.ndarray) -> float:
    if len(pred_list) < 2:
        return 0.0
    valid = [i for i in pred_list if i < item_embeddings.shape[0]]
    if len(valid) < 2:
        return 0.0
    embeds = item_embeddings[valid]
    total_dist = 0.0
    count = 0
    for i in range(len(embeds)):
        for j in range(i+1, len(embeds)):
            norm_i = np.linalg.norm(embeds[i])
            norm_j = np.linalg.norm(embeds[j])
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(embeds[i], embeds[j]) / (norm_i * norm_j)
                total_dist += 1 - sim
                count += 1
    return total_dist / count if count > 0 else 0.0

def compute_avg_diversity(recommended_lists: List[List[int]], item_embeddings: np.ndarray) -> float:
    if not recommended_lists:
        return 0.0
    divs = [compute_intra_list_diversity(lst, item_embeddings) for lst in recommended_lists]
    return np.mean(divs) if divs else 0.0

# --------------------------------------------------------------
# 6. Model size
# --------------------------------------------------------------
def get_model_size_mb(model) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

# --------------------------------------------------------------
# 7. Latency benchmark (single inference, average, p95, p99)
# --------------------------------------------------------------
def benchmark_latency(model, batch_data, device, num_iterations=100, warmup=10):
    """
    Measures average, p95, p99 latency (ms) for a single forward pass.
    batch_data: a tuple (items_t, events_t, hour_t, day_t, gap_t) ready for model.
    Returns dict with avg, p95, p99.
    """
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*batch_data)
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(*batch_data)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)
    return {
        'avg_ms': np.mean(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }

# --------------------------------------------------------------
# 8. CPU RAM usage
# --------------------------------------------------------------
def get_cpu_ram_usage_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024**2

# --------------------------------------------------------------
# 9. Unknown item ratio
# --------------------------------------------------------------
def compute_unknown_ratio(test_items: List[int], item_encoder: Dict) -> float:
    total = len(test_items)
    if total == 0:
        return 0.0
    unknown = sum(1 for it in test_items if it not in item_encoder)
    return unknown / total

# --------------------------------------------------------------
# 10. Session length analysis
# --------------------------------------------------------------
def evaluate_by_session_length(predictions_by_length: Dict[str, List[Dict]], k_values: List[int]) -> Dict:
    results = {}
    for length_group, preds in predictions_by_length.items():
        recall_list = []
        for p in preds:
            pred_k = p['preds'][:20]
            rec = 1.0 if p['target'] in pred_k else 0.0
            recall_list.append(rec)
        results[f'recall@20_{length_group}'] = np.mean(recall_list) if recall_list else 0.0
    return results

# --------------------------------------------------------------
# 11. User activity analysis
# --------------------------------------------------------------
def evaluate_by_user_activity(predictions_by_activity: Dict[str, List[Dict]], k_values: List[int]) -> Dict:
    results = {}
    for activity_group, preds in predictions_by_activity.items():
        recall_list = []
        for p in preds:
            pred_k = p['preds'][:20]
            rec = 1.0 if p['target'] in pred_k else 0.0
            recall_list.append(rec)
        results[f'recall@20_{activity_group}'] = np.mean(recall_list) if recall_list else 0.0
    return results

# --------------------------------------------------------------
# 12. Recommendation stability (Jaccard similarity)
# --------------------------------------------------------------
def compute_stability(consecutive_lists: List[List[int]]) -> float:
    if len(consecutive_lists) < 2:
        return 1.0
    sims = []
    for i in range(len(consecutive_lists)-1):
        set1 = set(consecutive_lists[i])
        set2 = set(consecutive_lists[i+1])
        if not set1 and not set2:
            sim = 1.0
        else:
            inter = len(set1 & set2)
            union = len(set1 | set2)
            sim = inter / union if union > 0 else 0.0
        sims.append(sim)
    return np.mean(sims) if sims else 0.0

# --------------------------------------------------------------
# 13. Business diversity (category level)
# --------------------------------------------------------------
def compute_category_diversity(pred_lists: List[List[int]], 
                               item_to_category: Dict[int, int]) -> float:
    if not pred_lists or not item_to_category:
        return 0.0
    unique_counts = []
    for lst in pred_lists:
        cats = {item_to_category.get(item) for item in lst if item in item_to_category}
        unique_counts.append(len(cats))
    return np.mean(unique_counts) if unique_counts else 0.0

def mmr_rerank(
    candidate_indices: List[int],
    candidate_scores: List[float],
    item_to_category: Dict[int, int],
    top_k: int = 20,
    lambda_: float = 0.5
) -> List[int]:
    """
    Re-rank candidates using Maximum Marginal Relevance (MMR).
    
    Balances relevance (model score) with diversity (category difference).
    
    Args:
        candidate_indices: List of item IDs (encoded) to consider
        candidate_scores: Corresponding relevance scores from the model
        item_to_category: Mapping item_id -> category_id (use 0 for unknown)
        top_k: Number of items to select
        lambda_: Weight for relevance (1-lambda_ is weight for diversity)
                0.3 = more diversity, 0.7 = more relevance
    
    Returns:
        Re-ranked list of top_k item IDs (decoded/encoded, same format as input)
    """
    if not candidate_indices or not item_to_category:
        return candidate_indices[:top_k]
    
    # Helper to get category, default to 0 (unique unknown category)
    def get_cat(item):
        return item_to_category.get(item, 0)
    
    selected = []
    remaining = list(candidate_indices)
    scores = {item: score for item, score in zip(candidate_indices, candidate_scores)}
    
    while len(selected) < top_k and remaining:
        best_mmr = -float('inf')
        best_idx = -1
        
        for i, item in enumerate(remaining):
            relevance = scores.get(item, 0.0)
            
            # Max similarity to already selected items (1 if same category, else 0)
            if selected:
                # Perfect similarity if same category, 0 otherwise
                max_sim = max(1.0 if get_cat(item) == get_cat(s) else 0.0 for s in selected)
            else:
                max_sim = 0.0
            
            # MMR = lambda * relevance - (1-lambda) * max_similarity
            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        
        if best_idx == -1:
            break
        
        selected.append(remaining.pop(best_idx))
    
    # Pad if we couldn't select enough (fallback)
    if len(selected) < top_k:
        selected.extend(remaining[:top_k - len(selected)])
    
    return selected
