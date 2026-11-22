"""Domain-Aware Class-Stratified Leakage-Safe Dataset Splitting Engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.generate_splits")


@dataclass
class SplitMetrics:
    """Statistical summary of an individual dataset split partition."""
    num_records: int
    record_percentage: float
    num_domains: int
    domain_percentage: float
    class_0_legitimate: int
    class_1_phishing: int
    phishing_ratio: float


@dataclass
class DatasetSplitsReport:
    """Master audit report detailing train, validation, and test partition metrics and leakage proof."""
    random_seed: int
    total_records: int
    total_domains: int
    train: SplitMetrics
    val: SplitMetrics
    test: SplitMetrics
    domain_leakage_detected: bool
    overlapping_domains_count: int
    record_conservation_verified: bool


class DomainAwareSplitter:
    """Partitions datasets strictly by registrable domain with class-aware stratification to guarantee ZERO cross-split data leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        assert round(train_ratio + val_ratio + test_ratio, 2) == 1.0, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    @staticmethod
    def verify_no_domain_leakage(
        train_domains: set[str],
        val_domains: set[str],
        test_domains: set[str],
    ) -> tuple[bool, set[str]]:
        """Mathematically prove that train, val, and test partitions share zero common domains."""
        train_val_overlap = train_domains.intersection(val_domains)
        train_test_overlap = train_domains.intersection(test_domains)
        val_test_overlap = val_domains.intersection(test_domains)

        all_leakages = train_val_overlap.union(train_test_overlap).union(val_test_overlap)
        has_leakage = len(all_leakages) > 0
        return not has_leakage, all_leakages

    def split_dataframe(
        self,
        df: pd.DataFrame,
        domain_column: str = "registrable_domain",
        label_column: str = "label",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DatasetSplitsReport]:
        """Perform class-stratified domain-level partitioning with deterministic random seed."""
        if domain_column not in df.columns:
            raise ValueError(f"Required domain column '{domain_column}' not found in dataframe")
        if label_column not in df.columns:
            raise ValueError(f"Required label column '{label_column}' not found in dataframe")

        df_work = df.copy()

        # Handle missing or empty registrable domains explicitly
        if df_work[domain_column].isna().any() or (df_work[domain_column].str.strip() == "").any():
            logger.warning("Found records with missing registrable_domain; applying synthetic domain keys to prevent loss")
            df_work[domain_column] = df_work.apply(
                lambda row: row[domain_column]
                if pd.notna(row.get(domain_column)) and str(row.get(domain_column)).strip()
                else f"__missing_domain_{row.name}__",
                axis=1,
            )

        # Aggregate record counts and labels per unique domain
        domain_stats = df_work.groupby(domain_column)[label_column].agg(
            total_records="count",
            phish_records="sum",
        ).reset_index()

        # Determine dominant class for domain-level stratification
        domain_stats["dominant_class"] = (
            domain_stats["phish_records"] > (domain_stats["total_records"] / 2.0)
        ).astype(int)

        np.random.seed(self.random_seed)

        train_dom_set: set[str] = set()
        val_dom_set: set[str] = set()
        test_dom_set: set[str] = set()

        # Stratified allocation across dominant classes
        for cls_val in [0, 1]:
            subset = domain_stats[domain_stats["dominant_class"] == cls_val].copy()
            if subset.empty:
                continue

            shuffled_indices = np.random.permutation(len(subset))
            shuffled = subset.iloc[shuffled_indices]

            n = len(shuffled)
            n_tr = int(n * self.train_ratio)
            n_va = int(n * self.val_ratio)

            train_dom_set.update(shuffled.iloc[:n_tr][domain_column])
            val_dom_set.update(shuffled.iloc[n_tr : n_tr + n_va][domain_column])
            test_dom_set.update(shuffled.iloc[n_tr + n_va :][domain_column])

        # Verify zero domain leakage
        no_leak, leakages = self.verify_no_domain_leakage(train_dom_set, val_dom_set, test_dom_set)
        if not no_leak:
            raise AssertionError(f"CRITICAL: Domain leakage detected across splits: {list(leakages)[:10]}")

        # Partition dataframe rows by domain membership
        df_train = df_work[df_work[domain_column].isin(train_dom_set)].copy()
        df_val = df_work[df_work[domain_column].isin(val_dom_set)].copy()
        df_test = df_work[df_work[domain_column].isin(test_dom_set)].copy()

        # Strict conservation assertion: every input record must appear in exactly one split
        total_input_len = len(df_work)
        total_split_len = len(df_train) + len(df_val) + len(df_test)
        if total_input_len != total_split_len:
            raise AssertionError(
                f"Record conservation failed: {total_input_len} input rows != {total_split_len} split rows"
            )

        union_indices = set(df_train.index).union(set(df_val.index)).union(set(df_test.index))
        if len(union_indices) != total_input_len:
            raise AssertionError("Duplicate or disjoint index mismatch across split partitions")

        total_domains_count = len(domain_stats)

        def _get_metrics(df_sub: pd.DataFrame, dom_set: set[str]) -> SplitMetrics:
            c0 = int((df_sub[label_column] == 0).sum())
            c1 = int((df_sub[label_column] == 1).sum())
            total = len(df_sub)
            ratio = round(c1 / max(total, 1), 4)
            return SplitMetrics(
                num_records=total,
                record_percentage=round((total / max(total_input_len, 1)) * 100, 2),
                num_domains=len(dom_set),
                domain_percentage=round((len(dom_set) / max(total_domains_count, 1)) * 100, 2),
                class_0_legitimate=c0,
                class_1_phishing=c1,
                phishing_ratio=ratio,
            )

        report = DatasetSplitsReport(
            random_seed=self.random_seed,
            total_records=total_input_len,
            total_domains=total_domains_count,
            train=_get_metrics(df_train, train_dom_set),
            val=_get_metrics(df_val, val_dom_set),
            test=_get_metrics(df_test, test_dom_set),
            domain_leakage_detected=not no_leak,
            overlapping_domains_count=len(leakages),
            record_conservation_verified=True,
        )

        logger.info(
            f"Class-aware domain split complete: "
            f"Train={len(df_train)} ({report.train.num_domains} doms), "
            f"Val={len(df_val)} ({report.val.num_domains} doms), "
            f"Test={len(df_test)} ({report.test.num_domains} doms) | Leakage=0"
        )
        return df_train, df_val, df_test, report

    @staticmethod
    def split_by_temporal_holdout(
        df: pd.DataFrame,
        timestamp_column: str = "collection_timestamp",
        test_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split records based on chronological timestamp ordering for temporal generalization testing."""
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found")

        df_valid_ts = df[df[timestamp_column].notna()].copy()
        df_valid_ts = df_valid_ts.sort_values(by=timestamp_column)

        cutoff_idx = int(len(df_valid_ts) * (1.0 - test_ratio))
        train_df = df_valid_ts.iloc[:cutoff_idx]
        test_df = df_valid_ts.iloc[cutoff_idx:]
        return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-Aware Dataset Splitting")
    parser.add_argument("--input-file", default="data/processed/urls/corpus_normalized.parquet")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        in_path = Path("data/processed/urls/corpus_normalized.csv")
    if not in_path.exists():
        logger.error(f"Input file not found at {in_path}. Run preprocess_pipeline.py first.")
        sys.exit(1)

    logger.info(f"Loading cleaned corpus from {in_path}")
    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    splitter = DomainAwareSplitter(random_seed=args.seed)
    df_train, df_val, df_test, report = splitter.split_dataframe(df)

    out_root = Path(args.splits_dir)
    for name, sub_df in [("train", df_train), ("validation", df_val), ("test", df_test)]:
        split_dir = out_root / name
        split_dir.mkdir(parents=True, exist_ok=True)
        sub_df.to_parquet(split_dir / "urls.parquet", index=False)
        sub_df.to_csv(split_dir / "urls.csv", index=False)
        logger.info(f"Saved {len(sub_df)} records to {split_dir}")

    meta_file = out_root / "split_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"Split metadata written to {meta_file}")

    print("\n" + "=" * 80)
    print("🛡️  DOMAIN-AWARE CLASS-STRATIFIED LEAKAGE-SAFE SPLITTING COMPLETE")
    print("=" * 80)
    print(f"• Fixed Random Seed       : {report.random_seed}")
    print(f"• Total Records Conserved : {report.total_records} across {report.total_domains} unique domains")
    print(
        f"• Train Split (70%)       : {report.train.num_records} URLs ({report.train.record_percentage}%) | "
        f"{report.train.num_domains} doms ({report.train.domain_percentage}%) | Legit: {report.train.class_0_legitimate}, Phish: {report.train.class_1_phishing}"
    )
    print(
        f"• Validation Split (15%)  : {report.val.num_records} URLs ({report.val.record_percentage}%) | "
        f"{report.val.num_domains} doms ({report.val.domain_percentage}%) | Legit: {report.val.class_0_legitimate}, Phish: {report.val.class_1_phishing}"
    )
    print(
        f"• Test Split (15%)        : {report.test.num_records} URLs ({report.test.record_percentage}%) | "
        f"{report.test.num_domains} doms ({report.test.domain_percentage}%) | Legit: {report.test.class_0_legitimate}, Phish: {report.test.class_1_phishing}"
    )
    print(f"• Domain Leakage Check    : {'✅ PASSED (0 Overlapping Domains)' if not report.domain_leakage_detected else '❌ FAILED'}")
    print(f"• Record Conservation     : {'✅ VERIFIED (Exact Count Matched)' if report.record_conservation_verified else '❌ FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
