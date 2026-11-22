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

    def test_class_aware_stratification_representation(self, mock_dataset_df):
        """Verifies both legitimate (0) and phishing (1) classes appear across all partitions."""
        splitter = DomainAwareSplitter(random_seed=42)
        tr, val, test, rep = splitter.split_dataframe(mock_dataset_df)

        assert rep.record_conservation_verified is True
        assert rep.overlapping_domains_count == 0
        assert rep.domain_leakage_detected is False

        # Check each split has both classes represented
        assert (tr["label"] == 0).sum() > 0
        assert (tr["label"] == 1).sum() > 0
        assert (val["label"] == 0).sum() > 0 or (test["label"] == 0).sum() > 0

    def test_missing_domains_handled_without_loss(self):
        """Missing registrable_domain strings must not drop rows or crash splitter."""
        data = [
            {"canonical_url": "http://1.1.1.1/a", "registrable_domain": None, "label": 1},
            {"canonical_url": "http://1.1.1.2/b", "registrable_domain": "", "label": 0},
            {"canonical_url": "http://valid.com/c", "registrable_domain": "valid.com", "label": 1},
            {"canonical_url": "http://valid2.com/d", "registrable_domain": "valid2.com", "label": 0},
            {"canonical_url": "http://valid3.com/e", "registrable_domain": "valid3.com", "label": 1},
        ]
        df = pd.DataFrame(data)
        splitter = DomainAwareSplitter(random_seed=42)
        tr, val, test, rep = splitter.split_dataframe(df)

        # Assert zero row loss
        assert len(tr) + len(val) + len(test) == len(df)
        assert rep.record_conservation_verified is True

    def test_highly_imbalanced_domains(self):
        """One massive domain with 200 records + 10 small domains with 2 records each."""
        data = []
        for i in range(200):
            data.append({
                "canonical_url": f"https://mega-domain.com/page_{i}",
                "registrable_domain": "mega-domain.com",
                "label": 1,
            })
        for d in range(10):
            for i in range(2):
                data.append({
                    "canonical_url": f"https://tiny-domain-{d}.com/page_{i}",
                    "registrable_domain": f"tiny-domain-{d}.com",
                    "label": 0,
                })
        df = pd.DataFrame(data)
        splitter = DomainAwareSplitter(random_seed=42)
        tr, val, test, rep = splitter.split_dataframe(df)

        # Zero leakage check
        train_doms = set(tr["registrable_domain"])
        val_doms = set(val["registrable_domain"])
        test_doms = set(test["registrable_domain"])

        assert len(train_doms.intersection(val_doms)) == 0
        assert len(train_doms.intersection(test_doms)) == 0
        assert len(val_doms.intersection(test_doms)) == 0
        assert len(tr) + len(val) + len(test) == 220

    def test_temporal_holdout_split(self):
        """Verifies chronological order separation where test partition strictly follows train partition."""
        data = [
            {"canonical_url": "https://a.com", "collection_timestamp": "2024-01-01T00:00:00Z"},
            {"canonical_url": "https://b.com", "collection_timestamp": "2024-01-02T00:00:00Z"},
            {"canonical_url": "https://c.com", "collection_timestamp": "2024-01-03T00:00:00Z"},
            {"canonical_url": "https://d.com", "collection_timestamp": "2024-01-04T00:00:00Z"},
            {"canonical_url": "https://e.com", "collection_timestamp": "2024-01-05T00:00:00Z"},
        ]
        df = pd.DataFrame(data)
        tr, test = DomainAwareSplitter.split_by_temporal_holdout(df, test_ratio=0.4)

        assert len(tr) == 3
        assert len(test) == 2
        assert max(tr["collection_timestamp"]) < min(test["collection_timestamp"])
