"""
data_validation.py - Comprehensive Data Validation for RetailRocket Dataset
=========================================================================
Validates all raw data files before preprocessing to catch issues early.

Usage:
    # Standalone validation
    python data_validation.py
    
    # Import and use in your pipeline
    from data_validation import validate_all_data
    results = validate_all_data()
    if not all(r.is_valid for r in results.values()):
        raise ValueError("Data validation failed!")

Files validated:
    - events.csv
    - item_properties_part1.csv
    - item_properties_part2.csv
    - category_tree.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Validation Result Class
# ============================================================================

class ValidationResult:
    """Store validation results with errors, warnings, and statistics."""
    
    def __init__(self, name: str):
        self.name = name
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.passed = True
    
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
    
    def add_stat(self, key: str, value: Any) -> None:
        self.stats[key] = value
    
    @property
    def is_valid(self) -> bool:
        return self.passed
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def __str__(self) -> str:
        lines = [f"\n{'='*70}"]
        lines.append(f"VALIDATION: {self.name}")
        lines.append(f"{'='*70}")
        
        # Stats
        for key, val in self.stats.items():
            if isinstance(val, (int, float)):
                if isinstance(val, int):
                    lines.append(f"  {key}: {val:,}")
                else:
                    lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")
        
        # Warnings
        if self.warnings:
            lines.append(f"\nWARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"  [WARNING] {w}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        
        # Errors
        if self.errors:
            lines.append(f"\nERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
            lines.append(f"\nStatus: FAILED")
        else:
            lines.append(f"\nStatus: PASSED")
        
        lines.append("="*70)
        return "\n".join(lines)


# ============================================================================
# 2. Validator Functions
# ============================================================================

def validate_events(df: pd.DataFrame, file_path: Path) -> ValidationResult:
    """
    Validate events.csv:
    - Required columns exist
    - No null visitorid or itemid
    - Valid event types (view, addtocart, transaction)
    - Positive visitorid and itemid
    - Timestamp range is reasonable
    - No duplicate rows
    """
    result = ValidationResult("events.csv")
    total_rows = len(df)
    result.add_stat("Total rows", total_rows)
    
    # 1. Required columns
    required = {"timestamp", "visitorid", "event", "itemid"}
    missing = required - set(df.columns)
    if missing:
        result.add_error(f"Missing required columns: {missing}")
        return result
    result.add_stat("Columns", list(df.columns))
    
    # 2. Null checks
    null_visitor = df["visitorid"].isnull().sum()
    null_item = df["itemid"].isnull().sum()
    if null_visitor > 0:
        result.add_error(f"Null visitorid values: {null_visitor:,}")
    if null_item > 0:
        result.add_error(f"Null itemid values: {null_item:,}")
    
    # 3. Data types
    if not pd.api.types.is_integer_dtype(df["visitorid"]):
        result.add_warning(f"visitorid is {df['visitorid'].dtype}, expected int")
    if not pd.api.types.is_integer_dtype(df["itemid"]):
        result.add_warning(f"itemid is {df['itemid'].dtype}, expected int")
    
    # 4. Valid event types
    ALLOWED_EVENTS = {"view", "addtocart", "transaction"}
    event_counts = df["event"].value_counts().to_dict()
    result.add_stat("Event distribution", event_counts)
    
    invalid_events = set(df["event"].unique()) - ALLOWED_EVENTS
    if invalid_events:
        result.add_error(f"Invalid event types: {invalid_events}")
    
    # 5. Positive IDs
    negative_visitor = (df["visitorid"] < 0).sum()
    negative_item = (df["itemid"] < 0).sum()
    if negative_visitor > 0:
        result.add_error(f"Negative visitor IDs: {negative_visitor:,}")
    if negative_item > 0:
        result.add_error(f"Negative item IDs: {negative_item:,}")
    
    # 6. Timestamp validation
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        result.add_stat("Time range", f"{min_ts} to {max_ts}")
        result.add_stat("Total days", (max_ts - min_ts).days)
        
        # Check for anomalies (RetailRocket data is from 2015)
        if min_ts < pd.Timestamp("2015-01-01"):
            result.add_warning(f"Data starts before 2015: {min_ts}")
        if max_ts > pd.Timestamp("2016-01-01"):
            result.add_warning(f"Data extends beyond 2016: {max_ts}")
    
    # 7. Duplicates
    dupes = df.duplicated(subset=["visitorid", "itemid", "event", "timestamp"]).sum()
    if dupes > 0:
        result.add_warning(f"Duplicate rows found: {dupes:,} (will be removed during preprocessing)")
    
    # 8. Null timestamps
    if df["timestamp"].isnull().any():
        result.add_error(f"Null timestamps: {df['timestamp'].isnull().sum():,}")
    
    # 9. Session gap analysis (pre-check)
    if len(df) > 1:
        sorted_df = df.sort_values(["visitorid", "timestamp"])
        gaps = sorted_df.groupby("visitorid")["timestamp"].diff().dt.total_seconds()
        if len(gaps.dropna()) > 0:
            median_gap = gaps.median()
            max_gap = gaps.max()
            result.add_stat("Median time gap (seconds)", f"{median_gap:.1f}")
            result.add_stat("Max time gap (seconds)", f"{max_gap:.1f}")
            
            if max_gap > 86400:  # 24 hours
                result.add_warning(f"Large time gap detected: {max_gap/3600:.1f} hours")
    
    return result


def validate_item_properties(df: pd.DataFrame, file_path: Path) -> ValidationResult:
    """
    Validate item_properties_part1.csv and part2.csv:
    - Required columns exist
    - No null itemid
    - Known property types exist
    - Categoryid property exists and has valid values
    """
    result = ValidationResult(f"item_properties ({file_path.name})")
    total_rows = len(df)
    result.add_stat("Total rows", total_rows)
    
    # 1. Required columns
    required = {"itemid", "property", "value"}
    missing = required - set(df.columns)
    if missing:
        result.add_error(f"Missing required columns: {missing}")
        return result
    
    # 2. Null checks
    null_itemid = df["itemid"].isnull().sum()
    null_property = df["property"].isnull().sum()
    null_value = df["value"].isnull().sum()
    
    if null_itemid > 0:
        result.add_error(f"Null item IDs: {null_itemid:,}")
    if null_property > 0:
        result.add_error(f"Null properties: {null_property:,}")
    if null_value > 0:
        result.add_warning(f"Null values: {null_value:,}")
    
    # 3. Known property types
    known_props = {"categoryid", "790", "808", "791"}  # 790=price, 808=brand, 791=availability
    unique_props = set(df["property"].unique())
    result.add_stat("Unique properties", len(unique_props))
    
    unknown_props = unique_props - known_props
    if unknown_props:
        # Show only first 10 to avoid clutter
        sample = list(unknown_props)[:10]
        result.add_warning(f"Unknown property types (first 10): {sample}")
        if len(unknown_props) > 10:
            result.add_warning(f"Plus {len(unknown_props) - 10} more unknown properties")
    
    # 4. Categoryid analysis
    has_category = "categoryid" in unique_props
    result.add_stat("Has category data", "Yes" if has_category else "No")
    
    if has_category:
        cat_df = df[df["property"] == "categoryid"]
        cat_items = cat_df["itemid"].nunique()
        result.add_stat("Items with category", cat_items)
        
        # Check for valid category values
        cat_values = pd.to_numeric(cat_df["value"], errors="coerce")
        invalid_cats = cat_values.isnull().sum()
        if invalid_cats > 0:
            result.add_warning(f"Invalid category values (non-numeric): {invalid_cats:,}")
        
        # Check for negative category IDs
        neg_cats = (cat_values < 0).sum()
        if neg_cats > 0:
            result.add_warning(f"Negative category IDs: {neg_cats:,}")
    
    # 5. Duplicates
    dupes = df.duplicated(subset=["itemid", "property"]).sum()
    if dupes > 0:
        result.add_warning(f"Duplicate (item, property) pairs: {dupes:,}")
    
    return result


def validate_category_tree(df: pd.DataFrame, file_path: Path) -> ValidationResult:
    """
    Validate category_tree.csv:
    - Required columns exist
    - No null category IDs
    - All parent IDs exist in the tree
    - No circular dependencies
    - Reasonable tree depth
    """
    result = ValidationResult("category_tree.csv")
    total_rows = len(df)
    result.add_stat("Total nodes", total_rows)
    
    # 1. Column detection (handle both naming conventions)
    if "categoryid" in df.columns:
        cat_col = "categoryid"
        par_col = "parentid"
    elif "category_id" in df.columns:
        cat_col = "category_id"
        par_col = "parent_id"
    else:
        result.add_error(f"Missing category ID column. Found: {list(df.columns)}")
        return result
    
    # 2. Null checks
    if df[cat_col].isnull().any():
        result.add_error(f"Null category IDs: {df[cat_col].isnull().sum():,}")
    
    if df[par_col].isnull().any():
        # Some datasets use NaN for root nodes
        null_parents = df[par_col].isnull().sum()
        result.add_warning(f"Null parent IDs (root nodes): {null_parents:,}")
    
    # 3. Root nodes
    roots = df[df[par_col] == -1]
    if len(roots) == 0:
        # Try NaN as root indicator
        roots = df[df[par_col].isnull()]
        if len(roots) > 0:
            result.add_warning(f"Root nodes use NaN (not -1). Found {len(roots):,} roots")
        else:
            result.add_warning("No root nodes found (parent_id = -1 or NaN)")
    else:
        result.add_stat("Root nodes", len(roots))
    
    if len(roots) > 100:
        result.add_warning(f"Unusually many root nodes: {len(roots):,}")
    
    # 4. Parent-child integrity
    parent_ids = set(df[par_col].dropna().unique()) - {-1}
    all_ids = set(df[cat_col].unique())
    missing_parents = parent_ids - all_ids
    
    if missing_parents:
        result.add_error(f"Parent IDs that don't exist as categories: {len(missing_parents):,}")
        # Show first 10 as sample
        sample = list(missing_parents)[:10]
        result.add_error(f"Sample missing parent IDs: {sample}")
    
    # 5. Detect cycles (simplified topological check)
    depth_map = {}
    max_depth = 0
    iter_count = 0
    max_iter = len(df) * 2  # Safety limit
    
    while len(depth_map) < len(df) and iter_count < max_iter:
        iter_count += 1
        changed = False
        
        for _, row in df.iterrows():
            cat = row[cat_col]
            par = row[par_col]
            
            # Skip if already processed
            if cat in depth_map:
                continue
                
            # Root node
            if par == -1 or pd.isna(par):
                depth_map[cat] = 0
                changed = True
            # Parent already processed
            elif par in depth_map:
                depth_map[cat] = depth_map[par] + 1
                max_depth = max(max_depth, depth_map[cat])
                changed = True
        
        if not changed:
            break
    
    result.add_stat("Max tree depth", max_depth)
    
    if max_depth > 20:
        result.add_warning(f"Very deep category tree: depth={max_depth}")
    
    # Check for nodes not reached (potential cycles or disconnected)
    unreached = len(df) - len(depth_map)
    if unreached > 0:
        result.add_error(f"Nodes not reachable from root: {unreached:,}")
        if unreached > 0:
            result.add_warning("Circular dependencies or disconnected tree detected")
    
    return result


# ============================================================================
# 3. Main Validation Function
# ============================================================================

def validate_all_data(data_dir: Optional[Path] = None) -> Dict[str, ValidationResult]:
    """
    Validate all raw data files in the data directory.
    
    Args:
        data_dir: Path to data/raw directory (defaults to project's data/raw)
    
    Returns:
        Dict of filename -> ValidationResult
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If any validation errors occur (after collecting all)
    """
    if data_dir is None:
        # Try to find the data directory relative to this file
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data" / "raw"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info(f"Validating data in: {data_dir}")
    
    # File mapping: filename -> (validator_func, required)
    files = {
        "events.csv": (validate_events, True),
        "item_properties_part1.csv": (validate_item_properties, True),
        "item_properties_part2.csv": (validate_item_properties, True),
        "category_tree.csv": (validate_category_tree, True),
    }
    
    results: Dict[str, ValidationResult] = {}
    all_valid = True
    total_errors = 0
    total_warnings = 0
    
    for filename, (validator, required) in files.items():
        file_path = data_dir / filename
        if not file_path.exists():
            if required:
                results[filename] = ValidationResult(filename)
                results[filename].add_error(f"File not found: {file_path}")
                all_valid = False
            continue
        
        try:
            # Load file with appropriate settings
            if filename == "events.csv":
                df = pd.read_csv(
                    file_path,
                    dtype={"visitorid": "int32", "itemid": "int32", "event": "category"},
                    engine="c"
                )
                # Convert timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            elif filename == "category_tree.csv":
                df = pd.read_csv(file_path, engine="c")
            else:  # item_properties files
                df = pd.read_csv(
                    file_path,
                    dtype={"itemid": "int32"},
                    engine="c"
                )
            
            results[filename] = validator(df, file_path)
            if not results[filename].is_valid:
                all_valid = False
                total_errors += len(results[filename].errors)
            total_warnings += len(results[filename].warnings)
            
        except Exception as e:
            results[filename] = ValidationResult(filename)
            results[filename].add_error(f"Failed to load/validate: {str(e)}")
            all_valid = False
    
    # Print all results
    print("\n" + "="*70)
    print("DATA VALIDATION SUMMARY")
    print("="*70)
    
    for result in results.values():
        print(result)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Files validated: {len(results)}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total warnings: {total_warnings}")
    
    if all_valid:
        logger.info("All data validation passed successfully.")
    else:
        logger.error(f"Validation failed with {total_errors} errors. Fix issues before proceeding.")
    
    return results


# ============================================================================
# 4. Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        results = validate_all_data()
        # Exit with error code if validation failed
        if not all(r.is_valid for r in results.values()):
            sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)