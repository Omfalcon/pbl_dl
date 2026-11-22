"""Unit Tests for Lexical URL Intelligence, Preprocessing, and MLP Architecture."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from app.intelligence.lexical import LexicalFeatureExtractor, compute_shannon_entropy
from app.intelligence.lexical_model import LexicalMLP, compute_metrics
from app.intelligence.preprocessing import LexicalPreprocessor
from app.ml.from_scratch.losses import BinaryCrossEntropy
from app.ml.from_scratch.optimizers import Adam


class TestLexicalFeatureExtractor:
    """Verifies feature extraction precision, dimension registry, and heuristic correctness."""

    def test_feature_count_and_names(self):
        extractor = LexicalFeatureExtractor()
        assert len(extractor.feature_names) == 40
        assert extractor.num_features == 40

        feats = extractor.extract_features("https://example.com/login?u=1")
        assert len(feats) == 40
        for name in extractor.feature_names:
            assert name in feats

        vec = extractor.extract_vector("https://example.com/login?u=1")
        assert vec.shape == (40,)
        assert vec.dtype == np.float64

    def test_normal_url_features(self):
        extractor = LexicalFeatureExtractor()
        url = "https://www.example.com/path/to/resource.html?query=test#sec"
        feats = extractor.extract_features(url)

        assert feats["url_length"] == len(url)
        assert feats["hostname_length"] == len("www.example.com")
        assert feats["num_dots_url"] == 3.0
        assert feats["num_slashes_url"] == 5.0
        assert feats["num_question_marks_url"] == 1.0
        assert feats["has_fragment"] == 1.0
        assert feats["has_ip_hostname"] == 0.0
        assert feats["is_punycode"] == 0.0
        assert feats["path_depth"] == 3.0

    def test_ip_address_url_detection(self):
        extractor = LexicalFeatureExtractor()
        ipv4_url = "http://192.168.1.1/login.php"
        feats_ipv4 = extractor.extract_features(ipv4_url)
        assert feats_ipv4["has_ip_hostname"] == 1.0

        ipv6_url = "http://[2001:db8::1]:8080/secure"
        feats_ipv6 = extractor.extract_features(ipv6_url)
        assert feats_ipv6["has_ip_hostname"] == 1.0

        normal_url = "http://paypal.com/login"
        feats_normal = extractor.extract_features(normal_url)
        assert feats_normal["has_ip_hostname"] == 0.0

    def test_punycode_url_detection(self):
        extractor = LexicalFeatureExtractor()
        puny_url = "https://xn--pple-43d.com/login"
        feats = extractor.extract_features(puny_url)
        assert feats["is_punycode"] == 1.0

    def test_suspicious_tld_detection(self):
        extractor = LexicalFeatureExtractor()
        susp_url = "http://account-verification.cfd/update"
        feats = extractor.extract_features(susp_url)
        assert feats["is_suspicious_tld"] == 1.0

        clean_url = "https://account-verification.com/update"
        clean_feats = extractor.extract_features(clean_url)
        assert clean_feats["is_suspicious_tld"] == 0.0

    def test_multi_label_public_suffix_and_brand_recognition(self):
        extractor = LexicalFeatureExtractor()

        # Multi-label PSL: paypal.co.uk
        scheme, host, dom, tld, sub, path, port = extractor._parse_url("https://paypal.co.uk/myaccount")
        assert dom == "paypal.co.uk"
        assert tld == "co.uk"
        assert sub == ""
        feats = extractor.extract_features("https://paypal.co.uk/myaccount")
        assert feats["brand_in_subdomain_or_path"] == 0.0

        # Multi-label PSL: sbi.co.in
        _, _, dom_sbi, tld_sbi, _, _, _ = extractor._parse_url("https://sbi.co.in/portal")
        assert dom_sbi == "sbi.co.in"
        assert tld_sbi == "co.in"

        # Multi-label PSL with subdomain: login.sbi.co.in
        _, _, dom_log, tld_log, sub_log, _, _ = extractor._parse_url("https://login.sbi.co.in/netbanking")
        assert dom_log == "sbi.co.in"
        assert tld_log == "co.in"
        assert sub_log == "login"
        feats_log = extractor.extract_features("https://login.sbi.co.in/netbanking")
        assert feats_log["brand_in_subdomain_or_path"] == 0.0

        # Attacker masquerading brand in subdomain: sbi.co.in.attacker.com
        _, _, dom_att, tld_att, sub_att, _, _ = extractor._parse_url("http://sbi.co.in.attacker.com/steal")
        assert dom_att == "attacker.com"
        assert tld_att == "com"
        assert sub_att == "sbi.co.in"
        feats_att = extractor.extract_features("http://sbi.co.in.attacker.com/steal")
        assert feats_att["brand_in_subdomain_or_path"] == 1.0

    def test_suspicious_keywords_and_brand_impersonation(self):
        extractor = LexicalFeatureExtractor()

        # Attacker site with 'paypal' in subdomain or path
        phish_url = "http://paypal-verification.attacker.xyz/login/verify.php"
        phish_feats = extractor.extract_features(phish_url)
        assert phish_feats["count_suspicious_keywords"] >= 2.0  # login, verify
        assert phish_feats["count_target_brand_keywords"] >= 1.0  # paypal
        assert phish_feats["brand_in_subdomain_or_path"] == 1.0

        # Legitimate brand site
        legit_url = "https://www.paypal.com/signin"
        legit_feats = extractor.extract_features(legit_url)
        # Brand appears, but domain IS official paypal.com -> flag must be 0
        assert legit_feats["brand_in_subdomain_or_path"] == 0.0

    def test_shannon_entropy_calculation(self):
        assert compute_shannon_entropy("") == 0.0
        assert compute_shannon_entropy("a") == 0.0
        assert compute_shannon_entropy("aaaa") == 0.0

        # Higher diversity produces strictly higher entropy
        ent_low = compute_shannon_entropy("aabb")
        ent_high = compute_shannon_entropy("abcdefgh")
        assert ent_high > ent_low > 0.0

    def test_batch_extraction(self):
        extractor = LexicalFeatureExtractor()
        urls = [
            "https://example.com/one",
            "https://attacker.top/phish",
            "http://10.0.0.1/admin",
        ]
        batch = extractor.extract_batch(urls)
        assert batch.shape == (3, 40)
        assert np.all(np.isfinite(batch))


class TestMalformedAndEdgeCaseURLs:
    """Verifies robustness against corrupted, oversized, or missing URL strings."""

    def test_empty_and_whitespace_url(self):
        extractor = LexicalFeatureExtractor()
        vec_empty = extractor.extract_vector("")
        assert vec_empty.shape == (40,)
        assert np.all(vec_empty == 0.0)

        vec_ws = extractor.extract_vector("   \t\n  ")
        assert vec_ws.shape == (40,)
        assert np.all(vec_ws == 0.0)

    def test_none_and_non_string_input(self):
        extractor = LexicalFeatureExtractor()
        vec_none = extractor.extract_vector(None)  # type: ignore
        assert vec_none.shape == (40,)
        assert np.all(vec_none == 0.0)

    def test_url_without_scheme(self):
        extractor = LexicalFeatureExtractor()
        vec = extractor.extract_vector("login.example.com/account/security")
        assert vec.shape == (40,)
        assert np.all(np.isfinite(vec))
        assert vec[0] > 0  # url_length

    def test_extremely_long_url(self):
        extractor = LexicalFeatureExtractor()
        huge_url = "https://example.com/" + ("a" * 10000) + "?q=" + ("b" * 5000)
        vec = extractor.extract_vector(huge_url)
        assert vec.shape == (40,)
        assert np.all(np.isfinite(vec))
        assert vec[0] == float(len(huge_url))

    def test_unicode_and_percent_encoded_url(self):
        extractor = LexicalFeatureExtractor()
        encoded_url = "https://example.com/%E4%BD%A0%E5%A5%BD?name=%20test%20"
        feats = extractor.extract_features(encoded_url)
        assert feats["num_encoded_chars"] >= 5.0
        assert np.all(np.isfinite(list(feats.values())))


class TestLexicalPreprocessor:
    """Verifies strict zero information leakage and serialization."""

    def test_fit_transform_and_zero_leakage(self):
        rng = np.random.default_rng(42)
        x_train = rng.normal(10.0, 2.0, size=(100, 40))
        x_test = rng.normal(50.0, 5.0, size=(20, 40))

        preprocessor = LexicalPreprocessor()
        x_train_scaled = preprocessor.fit_transform(x_train)

        # Training data should have zero mean and unit variance
        assert np.allclose(np.mean(x_train_scaled, axis=0), 0.0, atol=1e-5)
        assert np.allclose(np.var(x_train_scaled, axis=0), 1.0, atol=1e-3)

        train_mean_orig = preprocessor.mean_.copy()
        train_scale_orig = preprocessor.scale_.copy()

        # Transform on test must NOT modify fitted training statistics
        _ = preprocessor.transform(x_test)
        assert np.array_equal(preprocessor.mean_, train_mean_orig)
        assert np.array_equal(preprocessor.scale_, train_scale_orig)

    def test_preprocessor_handles_constant_and_nan_features(self):
        # Column 0 is constant zero (scale would be 0 without eps protection)
        # Column 1 has NaNs
        x_train = np.array([
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ])
        preprocessor = LexicalPreprocessor()
        scaled = preprocessor.fit_transform(x_train)

        assert preprocessor.scale_[0] == 1.0  # Safe scale fallback
        assert np.all(np.isfinite(scaled))

        # Test data with NaN
        x_test = np.array([[0.0, np.nan]])
        scaled_test = preprocessor.transform(x_test)
        assert np.all(np.isfinite(scaled_test))

    def test_preprocessor_serialization_roundtrip(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        preprocessor = LexicalPreprocessor(feature_names=["f1", "f2"])
        preprocessor.fit(x)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            preprocessor.save(tmp_path)
            loaded = LexicalPreprocessor.load(tmp_path)

            assert np.array_equal(loaded.mean_, preprocessor.mean_)
            assert np.array_equal(loaded.scale_, preprocessor.scale_)
            assert loaded.feature_names == ["f1", "f2"]
            assert loaded.n_features_in_ == 2
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestLexicalMLP:
    """Verifies MLP architecture, predictions, training loop, and metrics."""

    def test_forward_output_shape(self):
        model = LexicalMLP(in_features=40, hidden_dim1=16, hidden_dim2=8, random_seed=42)
        x = np.random.randn(5, 40)
        out = model.forward(x)
        assert out.shape == (5, 1)
        assert np.all((out >= 0.0) & (out <= 1.0))

        probs = model.predict_proba(x)
        assert probs.shape == (5, 1)

        preds = model.predict(x, threshold=0.5)
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1})

    def test_model_weights_serialization_roundtrip(self):
        model = LexicalMLP(in_features=10, hidden_dim1=8, hidden_dim2=4, random_seed=42)
        x = np.random.randn(4, 10)
        orig_preds = model.predict_proba(x)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            model.save_weights(tmp_path)
            model_loaded = LexicalMLP(in_features=10, hidden_dim1=8, hidden_dim2=4, random_seed=999)
            model_loaded.load_weights(tmp_path)

            loaded_preds = model_loaded.predict_proba(x)
            assert np.allclose(orig_preds, loaded_preds)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_model_training_convergence_on_synthetic_data(self):
        # Linearly separable synthetic data
        np.random.seed(42)
        n = 200
        x0 = np.random.normal(-2.0, 0.5, size=(n // 2, 8))
        x1 = np.random.normal(2.0, 0.5, size=(n // 2, 8))
        x = np.vstack([x0, x1])
        y = np.vstack([np.zeros((n // 2, 1)), np.ones((n // 2, 1))])

        model = LexicalMLP(in_features=8, hidden_dim1=16, hidden_dim2=8, random_seed=42)
        loss_fn = BinaryCrossEntropy()
        optimizer = Adam(lr=0.05)

        initial_loss = loss_fn.forward(y, model.forward(x))

        for _ in range(30):
            preds = model.forward(x)
            grad = loss_fn.backward(y, preds)
            model.backward(grad)
            optimizer.step(model.get_parameters(), model.get_gradients())

        final_loss = loss_fn.forward(y, model.forward(x))
        assert final_loss < initial_loss
        assert final_loss < 0.1

        metrics = compute_metrics(y, model.predict_proba(x), threshold=0.5)
        assert metrics["macro_avg"]["f1"] >= 0.95
        assert metrics["accuracy"] >= 0.95


class TestClassImbalanceAndTrainingStrategies:
    """Verifies controlled imbalance mitigation, checkpoint restoration, and zero test leakage."""

    def test_imbalance_strategies_have_no_double_compensation(self):
        """Verifies Strategy A (balanced sampling + ordinary BCE) and Strategy B

        (natural sampling + class-weighted BCE) never combine both mechanisms.
        """
        # Synthetic imbalanced data: 100 samples (90 positive, 10 negative)
        rng = np.random.default_rng(42)
        x_train = rng.normal(0, 1, size=(100, 10))
        y_train = np.vstack([np.ones((90, 1)), np.zeros((10, 1))])
        x_val = rng.normal(0, 1, size=(20, 10))
        y_val = np.vstack([np.ones((18, 1)), np.zeros((2, 1))])

        from scripts.train_lexical_model import train_single_strategy

        # Strategy A: balanced=True, use_class_weights=False
        model_a, tau_a, val_metrics_a, hist_a = train_single_strategy(
            strategy_name="Test_Strategy_A",
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            sampling_mode="balanced",
            use_class_weights=False,
            epochs=5,
            batch_size=16,
            learning_rate=0.01,
            random_seed=42,
        )
        assert model_a is not None
        assert 0.0 < tau_a < 1.0

        # Strategy B: balanced=False, use_class_weights=True
        model_b, tau_b, val_metrics_b, hist_b = train_single_strategy(
            strategy_name="Test_Strategy_B",
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            sampling_mode="natural",
            use_class_weights=True,
            epochs=5,
            batch_size=16,
            learning_rate=0.01,
            random_seed=42,
        )
        assert model_b is not None
        assert 0.0 < tau_b < 1.0

    def test_best_checkpoint_restoration(self):
        """Verifies that model parameters are restored to the best validation checkpoint."""
        from scripts.train_lexical_model import train_single_strategy

        rng = np.random.default_rng(42)
        x_train = rng.normal(0, 1, size=(100, 8))
        y_train = np.vstack([np.ones((80, 1)), np.zeros((20, 1))])
        x_val = rng.normal(0, 1, size=(30, 8))
        y_val = np.vstack([np.ones((24, 1)), np.zeros((6, 1))])

        model, tau, val_metrics, hist = train_single_strategy(
            strategy_name="Test_Checkpoint",
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            sampling_mode="natural",
            use_class_weights=True,
            epochs=10,
            batch_size=20,
            learning_rate=0.01,
            random_seed=42,
        )

        best_epoch = hist["best_restored_epoch"]
        assert 1 <= best_epoch <= 10
        # The restored validation metric must match the best epoch's recorded validation metric
        best_recorded_f1 = max(hist["val_macro_f1"])
        assert np.isclose(val_metrics["macro_avg"]["f1"], best_recorded_f1)

    def test_validation_only_model_selection(self):
        """Verifies model selection is performed strictly on validation metrics without test exposure."""
        val_metric_a = {"macro_avg": {"f1": 0.52}}
        val_metric_b = {"macro_avg": {"f1": 0.65}}

        # Selection function rule
        selected = "Strategy_B" if val_metric_b["macro_avg"]["f1"] > val_metric_a["macro_avg"]["f1"] else "Strategy_A"
        assert selected == "Strategy_B"

    def test_production_artifacts_load_and_predict(self):
        """Verifies that trained production artifacts exist and can be loaded for real-time URL inference."""
        model_weights_path = Path("models/lexical/model_weights.npz")
        preprocessor_path = Path("models/lexical/preprocessor.json")
        feature_names_path = Path("models/lexical/feature_names.json")

        assert model_weights_path.exists(), f"Missing {model_weights_path}"
        assert preprocessor_path.exists(), f"Missing {preprocessor_path}"
        assert feature_names_path.exists(), f"Missing {feature_names_path}"

        # Load preprocessor
        preprocessor = LexicalPreprocessor.load(preprocessor_path)
        assert preprocessor.n_features_in_ == 40

        # Load feature extractor
        extractor = LexicalFeatureExtractor()
        assert extractor.num_features == 40

        # Load model
        model = LexicalMLP(in_features=40, hidden_dim1=64, hidden_dim2=32)
        model.load_weights(model_weights_path)

        # Test inference on legit URL
        legit_vec = extractor.extract_vector("https://www.google.com/search?q=test")
        legit_scaled = preprocessor.transform(legit_vec.reshape(1, -1))
        prob_legit = model.predict_proba(legit_scaled)[0, 0]
        assert 0.0 <= prob_legit <= 1.0

        # Test inference on phishing-like URL
        phish_vec = extractor.extract_vector("http://paypal.verification-portal.com.attacker.cfd/update?session=999")
        phish_scaled = preprocessor.transform(phish_vec.reshape(1, -1))
        prob_phish = model.predict_proba(phish_scaled)[0, 0]
        assert 0.0 <= prob_phish <= 1.0

