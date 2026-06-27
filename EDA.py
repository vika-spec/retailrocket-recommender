"""
Exploratory Data Analysis for Retail Rocket Dataset
===================================================
This script performs comprehensive EDA on the Retail Rocket dataset,
covering data quality, user behavior, item popularity, session patterns,
and temporal dynamics.

Usage:
    python EDA.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ----------------------------------------------------------------------
# 1. Load and Prepare Data
# ----------------------------------------------------------------------

def load_data():
    """Load raw data files."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'raw'
    
    print(f"Loading data from: {data_dir}")
    
    events = pd.read_csv(data_dir / 'events.csv')
    events.columns = [c.strip().lower() for c in events.columns]
    events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
    
    props1 = pd.read_csv(data_dir / 'item_properties_part1.csv')
    props2 = pd.read_csv(data_dir / 'item_properties_part2.csv')
    props = pd.concat([props1, props2], ignore_index=True)
    props.columns = [c.strip().lower() for c in props.columns]
    
    categories = pd.read_csv(data_dir / 'category_tree.csv')
    categories.columns = [c.strip().lower() for c in categories.columns]
    
    return events, props, categories

events, props, categories = load_data()

# ----------------------------------------------------------------------
# 2. Data Overview
# ----------------------------------------------------------------------

print("\n" + "="*80)
print("RETAIL ROCKET DATASET EDA")
print("="*80)

print("\n[1] DATA OVERVIEW")
print("-"*40)
print(f"Events: {len(events):,}")
print(f"Users: {events['visitorid'].nunique():,}")
print(f"Items: {events['itemid'].nunique():,}")
print(f"Properties: {len(props):,}")
print(f"Categories: {len(categories):,}")

# Time range
print(f"\nTime range: {events['timestamp'].min()} to {events['timestamp'].max()}")
print(f"Total days: {(events['timestamp'].max() - events['timestamp'].min()).days:,}")

# Memory usage
print(f"\nMemory usage: {events.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ----------------------------------------------------------------------
# 3. Event Type Distribution
# ----------------------------------------------------------------------

print("\n[2] EVENT TYPE DISTRIBUTION")
print("-"*40)
event_counts = events['event'].value_counts()
print(event_counts)
if 'view' in event_counts and 'transaction' in event_counts:
    print(f"\nView to transaction ratio: {event_counts['view'] / event_counts['transaction']:.1f}:1")

# ----------------------------------------------------------------------
# 4. User Behavior Analysis
# ----------------------------------------------------------------------

print("\n[3] USER BEHAVIOR")
print("-"*40)

# Events per user
user_events = events.groupby('visitorid').size()
print(f"Events per user:")
print(f"  Mean: {user_events.mean():.1f}")
print(f"  Median: {user_events.median():.0f}")
print(f"  Min: {user_events.min():}")
print(f"  Max: {user_events.max():}")

# Active users (users with > 1 event)
active_users = (user_events > 1).sum()
print(f"\nActive users (>1 event): {active_users:,} ({active_users/len(user_events)*100:.1f}%)")

# Users with purchases
users_with_purchases = events[events['event'] == 'transaction']['visitorid'].nunique()
print(f"Users with purchases: {users_with_purchases:,} ({users_with_purchases/events['visitorid'].nunique()*100:.1f}%)")

# ----------------------------------------------------------------------
# 5. Item Popularity Analysis
# ----------------------------------------------------------------------

print("\n[4] ITEM POPULARITY")
print("-"*40)

# Item interactions
item_interactions = events.groupby('itemid').size()
print(f"Interactions per item:")
print(f"  Mean: {item_interactions.mean():.1f}")
print(f"  Median: {item_interactions.median():.0f}")
print(f"  Min: {item_interactions.min():}")
print(f"  Max: {item_interactions.max():}")

# Popular items
top_items = item_interactions.nlargest(10)
print(f"\nTop 10 most popular items:")
for idx, (item, count) in enumerate(top_items.items(), 1):
    print(f"  #{idx:2d}: Item {item:>6d} - {count:,} interactions")

# Long tail
print(f"\nItems with only 1 interaction: {(item_interactions == 1).sum():,} ({((item_interactions == 1).sum() / len(item_interactions))*100:.1f}%)")
print(f"Items with < 5 interactions: {(item_interactions < 5).sum():,} ({((item_interactions < 5).sum() / len(item_interactions))*100:.1f}%)")

# ----------------------------------------------------------------------
# 6. Temporal Analysis
# ----------------------------------------------------------------------

print("\n[5] TEMPORAL PATTERNS")
print("-"*40)

# Daily activity
events['date'] = events['timestamp'].dt.date
daily_events = events.groupby('date').size()
print(f"Average daily events: {daily_events.mean():.0f}")
print(f"Min daily events: {daily_events.min():,}")
print(f"Max daily events: {daily_events.max():,}")

# Hourly patterns
events['hour'] = events['timestamp'].dt.hour
hourly_events = events.groupby('hour').size()
peak_hour = hourly_events.idxmax()
print(f"\nPeak activity hour: {peak_hour}:00 ({hourly_events.max():,} events)")

# Weekly patterns
events['dayofweek'] = events['timestamp'].dt.dayofweek
weekly_events = events.groupby('dayofweek').size()
busiest_day = weekly_events.idxmax()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print(f"Busiest day: {days[busiest_day]} ({weekly_events.max():,} events)")

# ----------------------------------------------------------------------
# 7. Session Analysis (FIXED)
# ----------------------------------------------------------------------

print("\n[6] SESSION ANALYSIS")
print("-"*40)

# Session detection
SESSIONS_GAP = 3600  # 1 hour in seconds
events = events.sort_values(['visitorid', 'timestamp'])
events['time_gap'] = events.groupby('visitorid')['timestamp'].diff().dt.total_seconds()
events['new_session'] = events['time_gap'].isna() | (events['time_gap'] > SESSIONS_GAP)
events['session_id'] = events.groupby('visitorid')['new_session'].cumsum().astype(str)
events['session_id'] = events['visitorid'].astype(str) + '_' + events['session_id']

# Session statistics (FIXED: use session_id as string)
session_lengths = events.groupby('session_id').size()
total_sessions = len(session_lengths)
print(f"Total sessions: {total_sessions:,}")
print(f"Session length:")
print(f"  Mean: {session_lengths.mean():.1f}")
print(f"  Median: {session_lengths.median():.0f}")
print(f"  Min: {session_lengths.min():}")
print(f"  Max: {session_lengths.max():}")

# Single-item sessions
single_item_sessions = (session_lengths == 1).sum()
print(f"\nSingle-item sessions: {single_item_sessions:,} ({single_item_sessions/total_sessions*100:.1f}%)")

# Sessions per user
sessions_per_user = events.groupby('visitorid')['session_id'].nunique()
print(f"\nSessions per user:")
print(f"  Mean: {sessions_per_user.mean():.1f}")
print(f"  Median: {sessions_per_user.median():.0f}")
print(f"  Max: {sessions_per_user.max():}")

# ----------------------------------------------------------------------
# 8. Category Analysis (FIXED)
# ----------------------------------------------------------------------

print("\n[7] CATEGORY ANALYSIS")
print("-"*40)

# Extract category information from properties
if 'property' in props.columns and 'value' in props.columns:
    # Find category property
    cat_props = props[props['property'] == 'categoryid'][['itemid', 'value']]
    
    if len(cat_props) > 0:
        cat_props = cat_props.drop_duplicates('itemid')
        cat_props['categoryid'] = pd.to_numeric(cat_props['value'], errors='coerce')
        cat_props = cat_props.dropna(subset=['categoryid'])
        cat_props['categoryid'] = cat_props['categoryid'].astype(int)
        
        print(f"Found category information for {len(cat_props):,} items")
        
        # Merge with events
        events_with_cat = events.merge(cat_props[['itemid', 'categoryid']], on='itemid', how='left')
        events_with_cat['categoryid'] = events_with_cat['categoryid'].fillna(-1).astype(int)
        
        # Items with categories
        items_with_cat = events_with_cat[events_with_cat['categoryid'] != -1]['itemid'].nunique()
        print(f"Items with category in events: {items_with_cat:,} ({items_with_cat/events['itemid'].nunique()*100:.1f}%)")
        
        # Events with categories
        events_with_cat_count = len(events_with_cat[events_with_cat['categoryid'] != -1])
        print(f"Events with category: {events_with_cat_count:,} ({events_with_cat_count/len(events)*100:.1f}%)")
        
        # Top categories
        cat_counts = events_with_cat[events_with_cat['categoryid'] != -1].groupby('categoryid').size()
        top_cats = cat_counts.nlargest(10)
        print(f"\nTop 10 categories:")
        for idx, (cat, count) in enumerate(top_cats.items(), 1):
            print(f"  #{idx:2d}: Category {cat:>6d} - {count:,} interactions")
    else:
        print("No categoryid property found in data")
else:
    print("Property data not available in expected format")

# ----------------------------------------------------------------------
# 9. Plots
# ----------------------------------------------------------------------

print("\n[8] GENERATING PLOTS...")
print("-"*40)

# Create output directory
fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)

# Plot 1: Event distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Event types
event_counts.plot(kind='bar', ax=axes[0, 0], color=sns.color_palette("husl", 3))
axes[0, 0].set_title('Event Type Distribution')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=0)

# User events distribution
user_events[user_events <= 100].hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Events per User (≤100)')
axes[0, 1].set_xlabel('Number of Events')
axes[0, 1].set_ylabel('Frequency')

# Item popularity
item_interactions[item_interactions <= 10].hist(bins=50, ax=axes[0, 2])
axes[0, 2].set_title('Item Interactions (≤10)')
axes[0, 2].set_xlabel('Number of Interactions')
axes[0, 2].set_ylabel('Frequency')

# Hourly distribution
hourly_events.plot(kind='bar', ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Events by Hour')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Events')

# Daily distribution
daily_events_sampled = daily_events.iloc[::20]
daily_events_sampled.index = pd.to_datetime(daily_events_sampled.index)
daily_events_sampled = daily_events_sampled.resample('D').sum()
daily_events_sampled.plot(ax=axes[1, 1], color='green', alpha=0.7)
axes[1, 1].set_title('Daily Events (Sampled)')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Events')

# Session length distribution
session_lengths_trimmed = session_lengths[session_lengths <= 20]
if len(session_lengths_trimmed) > 0:
    session_lengths_trimmed.hist(bins=19, ax=axes[1, 2], color='coral', edgecolor='black')
else:
    session_lengths[session_lengths <= 50].hist(bins=20, ax=axes[1, 2], color='coral', edgecolor='black')
axes[1, 2].set_title('Session Length Distribution')
axes[1, 2].set_xlabel('Items per Session')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(fig_dir / 'retailrocket_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to: {fig_dir / 'retailrocket_eda_overview.png'}")

# Plot 2: Heatmap - Hour vs Day
pivot = pd.crosstab(events['hour'], events['dayofweek'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=False, cmap='YlOrRd', ax=ax)
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_title('Events by Hour and Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Hour of Day')
plt.tight_layout()
plt.savefig(fig_dir / 'retailrocket_eda_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to: {fig_dir / 'retailrocket_eda_heatmap.png'}")

# Plot 3: Event proportions
fig, ax = plt.subplots(figsize=(8, 6))
event_percentages = event_counts / len(events) * 100
event_percentages_sorted = event_percentages.sort_values(ascending=False)
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
event_percentages_sorted.plot(kind='pie', autopct='%1.1f%%', 
                              explode=[0.02, 0.02, 0.02], 
                              colors=colors[:len(event_percentages_sorted)], ax=ax)
ax.set_title('Event Type Proportions')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(fig_dir / 'retailrocket_eda_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to: {fig_dir / 'retailrocket_eda_pie.png'}")

# Plot 4: User engagement
fig, ax = plt.subplots(figsize=(10, 6))
user_events_sorted = user_events.sort_values(ascending=False)
user_events_sorted.head(1000).plot(ax=ax, color='purple', alpha=0.7)
ax.set_title('User Engagement Distribution (Top 1000 Users)')
ax.set_xlabel('User Rank')
ax.set_ylabel('Events')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(fig_dir / 'retailrocket_eda_user_engagement.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to: {fig_dir / 'retailrocket_eda_user_engagement.png'}")

# Plot 5: Item popularity
fig, ax = plt.subplots(figsize=(10, 6))
item_counts_sorted = item_interactions.sort_values(ascending=False)
item_counts_sorted.head(1000).plot(ax=ax, color='teal', alpha=0.7)
ax.set_title('Item Popularity Distribution (Top 1000 Items)')
ax.set_xlabel('Item Rank')
ax.set_ylabel('Interactions')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(fig_dir / 'retailrocket_eda_item_popularity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to: {fig_dir / 'retailrocket_eda_item_popularity.png'}")

# ----------------------------------------------------------------------
# 10. Summary Statistics
# ----------------------------------------------------------------------

print("\n[9] SUMMARY")
print("="*80)

# Dataset quality
print("Data Quality:")
print(f"  Events with null visitorid: {events['visitorid'].isna().sum():,}")
print(f"  Events with null itemid: {events['itemid'].isna().sum():,}")

# Sparsity
print(f"\nData Sparsity:")
user_item_pairs = events.groupby(['visitorid', 'itemid']).size().reset_index().shape[0]
density = user_item_pairs / (events['visitorid'].nunique() * events['itemid'].nunique())
print(f"  User-Item pairs: {user_item_pairs:,}")
print(f"  Density: {density:.4%}")

print(f"\nKey Insights:")
print(f"  • {len(events):,} events from {events['visitorid'].nunique():,} users across {events['itemid'].nunique():,} items")
if 'view' in event_counts and 'transaction' in event_counts:
    print(f"  • {event_counts['view'] / event_counts['transaction']:.1f}x more views than purchases")
print(f"  • Peak activity: {peak_hour}:00, busiest day: {days[busiest_day]}")
print(f"  • {single_item_sessions/total_sessions*100:.1f}% of sessions are single-item")
print(f"  • {(item_interactions < 5).sum() / len(item_interactions)*100:.1f}% of items are long-tail (<5 interactions)")
print(f"  • Average {sessions_per_user.mean():.1f} sessions per user")

print("\n" + "="*80)
print("EDA complete! Figures saved to 'figures/' directory.")
print("="*80)