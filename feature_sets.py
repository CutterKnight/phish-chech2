from urlfeatures import extract

"""Which features each model may use (this file is the student's own work).

Each feature set is a name and either None (= every feature) or a list of feature names.
train.py trains every set listed here and writes the results to models/metrics.json,
so adding a set here is enough. The names of the features are the ones shown in the
table of the Check page (and the ones of your own features from extra_features.py).
"""

_EXCLUDED = {"keywords", "shortener"}
_NO_KW_SHORTENER = [k for k in extract("https://example.com").keys() if k not in _EXCLUDED]

FEATURE_SETS = {
    # every feature, including the student's own
    "all": None,
    # only lengths: the simplest possible idea
    "lengths_only": ["url_length", "host_length", "path_length", "query_length"],
    # all features except keywords and shortener
    "no_kw_shortener": _NO_KW_SHORTENER,
}