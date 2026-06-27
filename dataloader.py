"""
dataloader.py – Batch generator for GRU4Rec evaluation.
Converts raw test sessions into padded batches ready for model inference.
"""

"""
dataloader.py – Pre‑compute prefixes and provide a Dataset + DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any


class GRU4RecEvalDataset(Dataset):
    def __init__(self, items, events, hours, days, gaps, targets,
                 target_orig, hist_lens, user_ids):
        self.items = items
        self.events = events
        self.hours = hours
        self.days = days
        self.gaps = gaps
        self.targets = targets
        self.target_orig = target_orig
        self.hist_lens = hist_lens
        self.user_ids = user_ids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.items[idx], self.events[idx], self.hours[idx],
                self.days[idx], self.gaps[idx], self.targets[idx],
                self.target_orig[idx], self.hist_lens[idx], self.user_ids[idx])


def create_eval_dataloader(
    test_sessions: List[Dict[str, Any]],
    item_encoder: Dict[int, int],
    max_seq_len: int,
    gap_max: float,
    batch_size: int = 2048,
    user_ids_list: Optional[List[int]] = None,
) -> Tuple[DataLoader, List[Dict], List[int]]:
    """
    Pre‑compute all prefixes and return a DataLoader.

    Returns:
        dataloader: DataLoader that yields batches of tensors.
        filtered_sessions: list of filtered session dicts (for later use).
        all_test_items_original: list of all original item IDs (for unknown ratio).
    """
    items_buf, events_buf = [], []
    hours_buf, days_buf, gaps_buf = [], [], []
    targets_buf, target_orig_buf = [], []
    hist_lens_buf, user_ids_buf = [], []
    filtered_sessions = []
    all_test_items_original = []

    if user_ids_list is not None:
        assert len(user_ids_list) == len(test_sessions)

    for idx, sess in enumerate(test_sessions):
        items = list(sess['item_seq'])
        events = list(sess.get('event_seq', []))
        hours = list(sess.get('hour_seq', []))
        days = list(sess.get('day_seq', []))
        gaps = list(sess.get('time_gap_seq', []))

        min_len = min(len(items), len(events), len(hours), len(days), len(gaps))
        if min_len < 2:
            continue
        items = items[:min_len]
        events = events[:min_len]
        hours = hours[:min_len]
        days = days[:min_len]
        gaps = gaps[:min_len]

        # Keep only known items
        filtered = []
        for it, ev, hr, dy, ga in zip(items, events, hours, days, gaps):
            if it in item_encoder:
                filtered.append((it, ev, hr, dy, ga))
        if len(filtered) < 2:
            continue
        items, events, hours, days, gaps = zip(*filtered)
        items, events, hours, days, gaps = list(items), list(events), list(hours), list(days), list(gaps)
        enc_items = [item_encoder[it] for it in items]

        filtered_sessions.append(sess)
        all_test_items_original.extend(items)

        user_id = user_ids_list[idx] if user_ids_list is not None else idx

        for t in range(1, len(enc_items)):
            hist_items = enc_items[:t]
            hist_events = events[:t]
            hist_hours = hours[:t]
            hist_days = days[:t]
            hist_gaps = gaps[:t]
            target = enc_items[t]
            target_orig = items[t]
            hist_len = len(hist_items)

            if len(hist_items) > max_seq_len:
                hist_items = hist_items[-max_seq_len:]
                hist_events = hist_events[-max_seq_len:]
                hist_hours = hist_hours[-max_seq_len:]
                hist_days = hist_days[-max_seq_len:]
                hist_gaps = hist_gaps[-max_seq_len:]

            pad_len = max_seq_len - len(hist_items)
            items_pad = [0] * pad_len + list(hist_items)
            events_pad = [0] * pad_len + list(hist_events)
            hours_pad = [0] * pad_len + list(hist_hours)
            days_pad = [0] * pad_len + list(hist_days)
            gaps_pad = [0] * pad_len + list(hist_gaps)

            non_zero = [i for i, v in enumerate(items_pad) if v != 0]
            if non_zero:
                last_idx = non_zero[-1]
                hour_val = hours_pad[last_idx] / 23.0
                day_val = days_pad[last_idx] / 6.0
                gap_val = np.log1p(gaps_pad[last_idx]) / gap_max if gaps_pad[last_idx] else 0.0
            else:
                hour_val = day_val = gap_val = 0.0

            items_buf.append(items_pad)
            events_buf.append(events_pad)
            hours_buf.append(hour_val)
            days_buf.append(day_val)
            gaps_buf.append(gap_val)
            targets_buf.append(target)
            target_orig_buf.append(target_orig)
            hist_lens_buf.append(hist_len)
            user_ids_buf.append(user_id)

    # Convert to numpy arrays
    items_arr = np.array(items_buf, dtype=np.int64)
    events_arr = np.array(events_buf, dtype=np.int64)
    hours_arr = np.array(hours_buf, dtype=np.float32).reshape(-1, 1)
    days_arr = np.array(days_buf, dtype=np.float32).reshape(-1, 1)
    gaps_arr = np.array(gaps_buf, dtype=np.float32).reshape(-1, 1)
    targets_arr = np.array(targets_buf, dtype=np.int64)
    target_orig_arr = np.array(target_orig_buf, dtype=np.int64)
    hist_lens_arr = np.array(hist_lens_buf, dtype=np.int64)
    user_ids_arr = np.array(user_ids_buf, dtype=np.int64)

    dataset = GRU4RecEvalDataset(
        items_arr, events_arr, hours_arr, days_arr, gaps_arr,
        targets_arr, target_orig_arr, hist_lens_arr, user_ids_arr
    )

    # DataLoader with num_workers=0 (single‑process)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,          # as requested
        pin_memory=True,
    )

    return dataloader, filtered_sessions, all_test_items_original