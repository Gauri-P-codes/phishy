import phishy  # noqa: F401
from phishy.features import FEATURES, parse_url


def test_punycode_detection():
    assert FEATURES["has_punycode"].fn(parse_url("http://xn--e1afmkfd.example/login")) == 1
    assert FEATURES["has_punycode"].fn(parse_url("https://example.com")) == 0


def test_nonstandard_port_detection():
    assert FEATURES["has_nonstandard_port"].fn(parse_url("http://example.com:8080/login")) == 1
    assert FEATURES["has_nonstandard_port"].fn(parse_url("https://example.com:443/login")) == 0


def test_shortener_detection():
    assert FEATURES["is_url_shortener"].fn(parse_url("https://bit.ly/abc")) == 1
    assert FEATURES["is_url_shortener"].fn(parse_url("https://example.com/abc")) == 0


def test_brand_impersonation_heuristic():
    assert FEATURES["brand_impersonation"].fn(parse_url("https://paypal.example.net/login")) == 1
    assert FEATURES["brand_impersonation"].fn(parse_url("https://paypal.com/login")) == 0


def test_metrics_report_has_class_balanced_fields():
    import numpy as np
    from phishy.train import metrics

    report = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]))
    for key in ("balanced_accuracy", "precision_phishing", "recall_phishing",
                "specificity_legitimate", "f1_phishing", "mcc",
                "false_positives", "false_negatives", "confusion_matrix"):
        assert key in report
