"""
Feature engineering with a plug-in registry.

To add a feature, write a function that takes a ParsedURL and returns a number,
and decorate it with @feature(). Nothing else needs to change.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List
from urllib.parse import urlparse

import pandas as pd
import tldextract

# Offline extractor: uses the bundled public-suffix snapshot, no network calls.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WWW_RE = re.compile(r"^www\d*\.", re.IGNORECASE)
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

SUSPICIOUS_WORDS = {
    "login", "verify", "secure", "account", "update", "bank", "pay",
    "signin", "confirm", "password", "wallet", "alert", "suspend",
    "unlock", "billing",
}


# ----------------------------------------------------------------- parsing
@dataclass
class ParsedURL:
    raw: str
    host: str       # lowercase, no www.
    path: str
    query: str
    subdomain: str
    domain: str
    suffix: str     # tld, e.g. "com", "co.uk"

    @property
    def full(self) -> str:
        """Normalised URL: no scheme, no www."""
        return self.host + self.path + (("?" + self.query) if self.query else "")

    @property
    def registered_domain(self) -> str:
        if self.suffix:
            return f"{self.domain}.{self.suffix}"
        return self.domain or self.host


def parse_url(url: str) -> ParsedURL:
    """Strip scheme and www so formatting can't leak the label."""
    raw = str(url).strip()
    s = _SCHEME_RE.sub("", raw)
    try:
        p = urlparse("//" + s)
        host = _WWW_RE.sub("", (p.hostname or "").lower())
        path, query = p.path or "", p.query or ""
    except ValueError:
        host, path, query = "", "", ""
    ext = _EXTRACT(host)
    return ParsedURL(raw, host, path, query, ext.subdomain, ext.domain, ext.suffix)


# ---------------------------------------------------------------- registry
@dataclass
class Feature:
    name: str
    fn: Callable[[ParsedURL], float]
    network: bool = False  # True -> needs internet (WHOIS, HTTP...), off by default


FEATURES: Dict[str, Feature] = {}


def feature(name: str = None, network: bool = False):
    def deco(fn):
        FEATURES[name or fn.__name__] = Feature(name or fn.__name__, fn, network)
        return fn
    return deco


def active_features(include_network: bool = False) -> List[Feature]:
    return [f for f in FEATURES.values() if include_network or not f.network]


# --------------------------------------------------------------- helpers
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


# -------------------------------------------------------- built-in features
@feature()
def url_length(p): return len(p.full)

@feature()
def host_length(p): return len(p.host)

@feature()
def path_length(p): return len(p.path)

@feature()
def query_length(p): return len(p.query)

@feature()
def host_dots(p): return p.host.count(".")

@feature()
def num_subdomains(p): return len(p.subdomain.split(".")) if p.subdomain else 0

@feature()
def host_hyphens(p): return p.host.count("-")

@feature()
def host_digits(p): return sum(c.isdigit() for c in p.host)

@feature()
def has_ip(p): return 1 if _IP_RE.match(p.host) else 0

@feature()
def has_punycode(p): return 1 if "xn--" in p.host else 0

@feature()
def has_at_symbol(p): return 1 if "@" in p.raw else 0

@feature()
def has_encoded_chars(p): return 1 if re.search(r"%[0-9a-fA-F]{2}", p.full) else 0

@feature()
def has_nonstandard_port(p):
    try:
        raw = p.raw if "://" in p.raw else "//" + p.raw
        port = urlparse(raw).port
        return 1 if port is not None and port not in (80, 443) else 0
    except ValueError:
        return 1

@feature()
def is_url_shortener(p):
    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
                  "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at"}
    return 1 if p.registered_domain.lower() in shorteners else 0

@feature()
def brand_impersonation(p):
    brands = {"paypal", "apple", "microsoft", "google", "amazon", "facebook",
              "instagram", "netflix", "whatsapp", "linkedin", "dropbox", "adobe",
              "sbi", "hdfc", "icici", "axisbank"}
    text = " ".join([p.subdomain, p.path, p.query]).lower()
    tokens_found = set(tokens(text))
    return 1 if tokens_found.intersection(brands) and p.registered_domain.split(".")[0].lower() not in brands else 0

@feature()
def num_subdirs(p): return len([s for s in p.path.split("/") if s])

@feature()
def num_params(p): return len(p.query.split("&")) if p.query else 0

@feature()
def special_char_count(p): return sum(p.full.count(c) for c in "@%=?&")

@feature()
def digit_ratio(p):
    digits = sum(c.isdigit() for c in p.full)
    return digits / max(len(p.full), 1)

@feature()
def suspicious_words(p): return sum(t in SUSPICIOUS_WORDS for t in tokens(p.full))

@feature()
def host_entropy(p): return shannon_entropy(p.host)

@feature()
def path_entropy(p): return shannon_entropy(p.path + p.query)


# ------------------------------------------------------------ the builder
class FeatureBuilder:
    """Turns URLs into a model-ready DataFrame.

    Holds the one piece of learned state (TLD frequency table, fit on TRAIN only)
    and the exact column order, so training and prediction can never drift apart.
    """

    def __init__(self, include_network: bool = False):
        self.include_network = include_network
        self.tld_freq: Dict[str, float] = {}
        self.columns: List[str] = []

    def _raw(self, urls: Iterable[str], names: List[str]) -> pd.DataFrame:
        rows = []
        for u in urls:
            p = parse_url(u)
            row = {n: FEATURES[n].fn(p) for n in names}
            row["_tld"] = p.suffix or "unknown"
            rows.append(row)
        return pd.DataFrame(rows)

    def fit_transform(self, urls: Iterable[str]) -> pd.DataFrame:
        names = [f.name for f in active_features(self.include_network)]
        raw = self._raw(urls, names)
        self.tld_freq = raw["_tld"].value_counts(normalize=True).to_dict()
        self.columns = names + ["tld_freq"]
        return self._finish(raw)

    def transform(self, urls: Iterable[str]) -> pd.DataFrame:
        raw = self._raw(urls, [c for c in self.columns if c != "tld_freq"])
        return self._finish(raw)

    def _finish(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw["tld_freq"] = raw.pop("_tld").map(lambda t: self.tld_freq.get(t, 0.0))
        return raw[self.columns]
