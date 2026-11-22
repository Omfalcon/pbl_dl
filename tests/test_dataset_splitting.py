"""Unit Tests for Domain-Aware Leakage-Safe Dataset Splitting & Statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.datasets.generate_splits import DomainAwareSplitter


@pytest.fixture
def mock_dataset_df():
    """Generates a small DataFrame with multiple URLs sharing identical domains."""
    data = []
    # 10 domains, 4 URLs per domain (40 rows total)
    for i in range(1, 11):
        dom = f"domain{i}.com"
        label = 1 if i % 2 == 0 else 0
        for page in ["home", "login", "about", "contact"]:
            data.append({
                "canonical_url": f"https://{dom}/{page}",
                "registrable_domain": dom,
                "label": label,
                "source": "mock",
            })
    return pd.DataFrame(data)


class TestDomainAwareSplitting:
    """Verifies that splits are strictly partitioned by domain with zero data leakage."""

    def test_reproducibility_with_seed(self, mock_dataset_df):
        splitter = DomainAwareSplitter(random_seed=42)
        tr1, val1, test1, rep1 = splitter.split_dataframe(mock_dataset_df)

        splitter2 = DomainAwareSplitter(random_seed=42)
        tr2, val2, test2, rep2 = splitter2.split_dataframe(mock_dataset_df)

        assert tr1["canonical_url"].tolist() == tr2["canonical_url"].tolist()
        assert val1["canonical_url"].tolist() == val2["canonical_url"].tolist()
        assert test1["canonical_url"].tolist() == test2["canonical_url"].tolist()

    def test_same_domain_records_kept_in_single_split(self, mock_dataset_df):
        splitter = DomainAwareSplitter(random_seed=42)
        tr, val, test, rep = splitter.split_dataframe(mock_dataset_df)

        train_doms = set(tr["registrable_domain"])
        val_doms = set(val["registrable_domain"])
        test_doms = set(test["registrable_domain"])

        # Check total domains preserved
        assert len(train_doms) + len(val_doms) + len(test_doms) == 10

        # Check disjointness (ZERO leakage)
        assert len(train_doms.intersection(val_doms)) == 0
        assert len(train_doms.intersection(test_doms)) == 0
        assert len(val_doms.intersection(test_doms)) == 0

    def test_leakage_detector_catches_artificial_leakage(self):
        train_doms = {"google.com", "example.com", "paypal.com"}
        val_doms = {"microsoft.com", "apple.com"}
        test_doms = {"netflix.com", "paypal.com"}  # paypal.com is leaked from train!

        no_leak, overlaps = DomainAwareSplitter.verify_no_domain_leakage(
            train_doms, val_doms, test_doms
        )
        assert no_leak is False
        assert "paypal.com" in overlaps

    def test_zero_leakage_passes_cleanly(self):
        train_doms = {"google.com", "example.com"}
        val_doms = {"microsoft.com", "apple.com"}
        test_doms = {"netflix.com", "chase.com"}

        no_leak, overlaps = DomainAwareSplitter.verify_no_domain_leakage(
            train_doms, val_doms, test_doms
        )
        assert no_leak is True
        assert len(overlaps) == 0
