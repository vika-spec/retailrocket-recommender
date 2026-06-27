import numpy as np
import logging
from typing import List, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Basic ranking metrics
# ----------------------------------------------------------------------

def recall_at_k(preds: List[int], actual: int, k: int) -> float:
    return 1.0 if actual in preds[:k] else 0.0

def mrr_at_k(preds: List[int], actual: int, k: int) -> float:
    for i, item in enumerate(preds[:k]):
        if item == actual:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(preds: List[int], actual: int, k: int) -> float:
    if actual in preds[:k]:
        rank = preds[:k].index(actual) + 1
        dcg = 1.0 / np.log2(rank + 1)
        idcg = 1.0 / np.log2(2)
        return dcg / idcg
    return 0.0

def hitrate_at_k(preds: List[int], actual: int, k: int) -> float:
    return 1.0 if actual in preds[:k] else 0.0

def coverage_at_k(predictions: List[List[int]], total_items: int, k: int) -> float:
    unique_recommended = set()
    for pred in predictions:
        unique_recommended.update(pred[:k])
    return len(unique_recommended) / total_items if total_items > 0 else 0.0


# ----------------------------------------------------------------------
# Popularity baseline (academic mode)
# ----------------------------------------------------------------------

class PopularityBaseline:
    def __init__(self, item_counts: Dict[int, int], top_k: int = 20):
        self.top_items = sorted(
            item_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        self.top_items = [item for item, _ in self.top_items]

    def predict(self, num_recommendations: int = 20) -> List[int]:
        return self.top_items[:num_recommendations]

    def evaluate(self, test_sessions: List[Dict], item_encoder: Dict, k_values: List[int]) -> Dict:
        logger.info("Evaluating Popularity Baseline (academic mode)...")
        results = {k: {'recall': [], 'mrr': [], 'ndcg': [], 'hitrate': []} for k in k_values}
        recommended_items = set()
        max_k = max(k_values)

        for sess in tqdm(test_sessions, desc="Popularity Baseline"):
            items = list(sess['item_seq'])
            enc_items = [item_encoder[it] for it in items if it in item_encoder]
            if len(enc_items) < 2:
                continue

            for t in range(1, len(enc_items)):
                target = enc_items[t]
                pred_orig = self.predict(max_k)
                pred_enc = [item_encoder[it] for it in pred_orig if it in item_encoder]
                recommended_items.update(pred_enc[:20])

                for k in k_values:
                    pred_k = pred_enc[:k]
                    results[k]['recall'].append(recall_at_k(pred_k, target, k))
                    results[k]['mrr'].append(mrr_at_k(pred_k, target, k))
                    results[k]['ndcg'].append(ndcg_at_k(pred_k, target, k))
                    results[k]['hitrate'].append(hitrate_at_k(pred_k, target, k))

        final = {}
        for k in k_values:
            final[f'recall@{k}'] = np.mean(results[k]['recall']) if results[k]['recall'] else 0.0
            final[f'mrr@{k}'] = np.mean(results[k]['mrr']) if results[k]['mrr'] else 0.0
            final[f'ndcg@{k}'] = np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0.0
            final[f'hitrate@{k}'] = np.mean(results[k]['hitrate']) if results[k]['hitrate'] else 0.0

        final['coverage@20'] = len(recommended_items) / len(item_encoder) if item_encoder else 0.0
        return final