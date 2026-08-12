import math, re, json, os, string
from collections import Counter, defaultdict
import numpy as np
import joblib

_MODEL = None
_SCALER = None
_META = None         
_NGRAM = None        

def _load_artifacts():
    global _MODEL, _SCALER, _META, _NGRAM
    if _MODEL is None:
        _MODEL = joblib.load("models/model.pkl")
    if _SCALER is None:
        _SCALER = joblib.load("models/scaler.pkl")
    if _META is None:
        meta_path = "models/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                _META = json.load(f)
        else:
            _META = {"special_chars": list("!@#$%^&*()-_=+[]{}|;:'\",.<>/?`~\\ "), "feature_names": []}
    if _NGRAM is None and os.path.exists("models/ngram.pkl"):
        _NGRAM = joblib.load("models/ngram.pkl")
_QWERTY_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;\'",
    "zxcvbnm,./"
]
_QWERTY_ROWS += [row[::-1] for row in _QWERTY_ROWS]  

_COMMON_WORDS = {
    "password","pass","admin","welcome","letmein","qwerty","iloveyou","monkey",
    "dragon","football","baseball","abc123","login","starwars","princess"
}
_LEET_MAP = str.maketrans({"0":"o","1":"l","3":"e","4":"a","5":"s","7":"t","8":"b","9":"g","2":"z","$":"s","@":"a","!":"i"})

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    L = len(s)
    freq = [s.count(ch) / L for ch in set(s)]
    return -sum(p * math.log2(p) for p in freq)

def _charclass_bits(s: str, specials: set) -> float:
    if not s: return 0.0
    charset = 0
    if any(c.islower() for c in s): charset += 26
    if any(c.isupper() for c in s): charset += 26
    if any(c.isdigit() for c in s): charset += 10
    if any(not c.isalnum() for c in s):
        charset += max(1, len(specials))
    return len(s) * math.log2(max(1, charset))

def _ascii_seq_max(s: str) -> int:
    m = 1; cur = 1
    for i in range(1, len(s)):
        step = ord(s[i]) - ord(s[i-1])
        if step in (1, -1):
            cur += 1
            m = max(m, cur)
        else:
            cur = 1
    return m

def _qwerty_run_max(s: str) -> int:
    s_lower = s.lower()
    best = 1
    for row in _QWERTY_ROWS:
        for k in range(len(s_lower)):
            cur = 1
            for j in range(k+1, len(s_lower)):
                if s_lower[j-1:j+1] in row:
                    cur += 1
                    best = max(best, cur)
                else:
                    break
    return best

def _repeat_longest_run(s: str) -> int:
    longest = 1; run = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            run += 1; longest = max(longest, run)
        else:
            run = 1
    return longest

def _repeated_substring_fraction(s: str) -> float:
    for w in range(1, len(s)//2 + 1):
        if len(s) % w == 0:
            unit = s[:w]
            if unit * (len(s)//w) == s:
                return 1.0
    return 0.0

def _lz_complexity(s: str) -> int:
    seen = set(); w = ""; c = 0
    for ch in s:
        w += ch
        if w not in seen:
            seen.add(w); c += 1; w = ""
    return c

def _avg_bigram_surprisal(s: str, ngram) -> float:
    if not s or ngram is None or len(s) < 2: return 0.0
    char_counts = ngram["char_counts"]
    bigram_counts = ngram["bigram_counts"]
    vocab = max(1, ngram["vocab"])
    k = 1  
    total = 0.0; n = 0
    for i in range(len(s) - 1):
        c1, c2 = s[i], s[i+1]
        count = bigram_counts.get((c1, c2), 0)
        base = char_counts.get(c1, 0)
        prob = (count + k) / (base + k * vocab)
        total += -math.log2(prob); n += 1
    return total / n if n else 0.0

def _pattern_flags(s: str) -> dict:
    flags = {}
    flags["has_year"] = 1 if re.search(r"(19|20)\d{2}", s) else 0
    flags["has_date"] = 1 if re.search(r"\b(?:\d{2}[-/_]\d{2}[-/_]\d{2,4}|\d{4}[-/_]\d{2}[-/_]\d{2})\b", s) else 0
    flags["starts_with_digits"] = 1 if re.match(r"^\d{2,}", s) else 0
    flags["ends_with_digits"] = 1 if re.search(r"\d{2,}$", s) else 0
    s_norm = s.translate(_LEET_MAP).lower()
    flags["has_common_word"] = 1 if any(w in s_norm for w in _COMMON_WORDS) else 0
    return flags

def _counts(s: str):
    return (
        sum(c.isdigit() for c in s),
        sum(c.isupper() for c in s),
        sum(c.islower() for c in s),
        sum((not c.isalnum()) for c in s),
    )

FEATURE_NAMES = [
    "length","digits","uppers","lowers","specials",
    "shannon_entropy","charclass_bits",
    "unique_ratio","upper_ratio","lower_ratio","digit_ratio","special_ratio",
    "ascii_seq_max","qwerty_run_max","repeat_longest","repeat_sub_fraction",
    "lz_complexity","lz_norm","avg_bigram_surprisal",
    "has_year","has_date","starts_with_digits","ends_with_digits","has_common_word",
    "variety_classes"  
]

def extract_features(password: str):
    """Return np.array shape (1, n_features)."""
    _load_artifacts()
    specials = set(_META.get("special_chars", []))
    ngram = _NGRAM

    s = password or ""
    length = len(s)
    digits, uppers, lowers, specials_cnt = _counts(s)

    shannon = _shannon_entropy(s)
    bits = _charclass_bits(s, specials)
    unique_ratio = len(set(s)) / length if length else 0.0

    denom = max(1, length)
    upper_ratio = uppers / denom
    lower_ratio = lowers / denom
    digit_ratio = digits / denom
    special_ratio = specials_cnt / denom

    ascii_seq = _ascii_seq_max(s)
    qwerty_seq = _qwerty_run_max(s)
    repeat_longest = _repeat_longest_run(s)
    repeat_frac = _repeated_substring_fraction(s)

    lz = _lz_complexity(s)
    lz_norm = lz / denom

    surprisal = _avg_bigram_surprisal(s, ngram)
    flags = _pattern_flags(s)
    variety = sum([digits > 0, uppers > 0, lowers > 0, specials_cnt > 0])

    vec = np.array([
        length, digits, uppers, lowers, specials_cnt,
        shannon, bits,
        unique_ratio, upper_ratio, lower_ratio, digit_ratio, special_ratio,
        ascii_seq, qwerty_seq, repeat_longest, repeat_frac,
        lz, lz_norm, surprisal,
        flags["has_year"], flags["has_date"], flags["starts_with_digits"], flags["ends_with_digits"], flags["has_common_word"],
        variety
    ], dtype=float).reshape(1, -1)

    return vec

def _heuristic_weak(password: str) -> bool:
    """Force-downgrade obviously weak passwords regardless of model length bias.

    The training data separates classes almost purely by length, so the model
    tends to call any long password "Strong".  These rules catch the classic
    weak patterns the dataset does not express well.
    """
    if not password:
        return True
    s = password
    L = len(s)
    # Too short
    if L < 8:
        return True
    # All same character
    if len(set(s)) == 1:
        return True
    # Repeated run of the same character (aaa, 111, ...)
    if _repeat_longest_run(s) >= 3:
        return True
    # Fully repeated substring (abcabcabc, qwerty123qwerty123, ...)
    if _repeated_substring_fraction(s) == 1.0:
        return True
    # Long QWERTY keyboard sequence
    if _qwerty_run_max(s) >= 4:
        return True
    # Long ASCII/sequential pattern (123456..., abcdefg...)
    if _ascii_seq_max(s) >= 4:
        return True
    # Contains a common dictionary / breached word (leet-speak aware)
    s_norm = s.translate(_LEET_MAP).lower()
    if any(w in s_norm for w in _COMMON_WORDS):
        return True
    return False


def _heuristic_strong(password: str) -> bool:
    """Force-upgrade genuinely strong random passwords.

    The training data is length-dominated, so a short-but-random password with
    all four character classes gets under-rated as "Medium".  These rules
    recognise high-quality random passwords.
    """
    s = password or ""
    if len(s) < 10:
        return False
    digits = sum(c.isdigit() for c in s)
    uppers = sum(c.isupper() for c in s)
    lowers = sum(c.islower() for c in s)
    specials = sum(not c.isalnum() for c in s)
    # All four classes present, each at least once, and long enough
    if all([digits > 0, uppers > 0, lowers > 0, specials > 0]) and len(s) >= 12:
        return True
    # Very high Shannon entropy + good length, no weak patterns
    if len(s) >= 14 and _shannon_entropy(s) >= 3.4:
        return True
    return False


def predict_strength(password: str):
    """Returns label string."""
    _load_artifacts()
    if _heuristic_weak(password):
        return "Weak"
    if _heuristic_strong(password):
        return "Strong"
    x = extract_features(password)
    xs = _SCALER.transform(x)
    pred = int(_MODEL.predict(xs)[0])
    label_map = {0:"Weak", 1:"Medium", 2:"Strong"}
    return label_map.get(pred, "Unknown")

def explain_password(password: str) -> dict:
    """Detailed features + effective entropy + crack time estimates."""
    _load_artifacts()
    x = extract_features(password)
    xs = _SCALER.transform(x)
    proba = _MODEL.predict_proba(xs)[0].tolist() if hasattr(_MODEL, "predict_proba") else None
    label = predict_strength(password)
    force_weak = _heuristic_weak(password)
    force_strong = _heuristic_strong(password)
    if force_weak:
        label = "Weak"
        if proba is not None:
            proba = [1.0, 0.0, 0.0]
    elif force_strong:
        label = "Strong"
        if proba is not None:
            proba = [0.0, 0.0, 1.0]

    s = password or ""
    specials = set(_META.get("special_chars", []))
    base_bits = _charclass_bits(s, specials)
    penalties = 0.0

    ascii_seq = _ascii_seq_max(s)
    qwerty_seq = _qwerty_run_max(s)
    repeat_longest = _repeat_longest_run(s)
    repeat_frac = _repeated_substring_fraction(s)
    flags = _pattern_flags(s)

    penalties += max(0, (ascii_seq - 2)) * 1.5
    penalties += max(0, (qwerty_seq - 2)) * 2.0
    penalties += max(0, (repeat_longest - 2)) * 1.0
    penalties += repeat_frac * 8.0
    penalties += 6.0 * flags["has_common_word"]
    penalties += 4.0 * flags["has_year"] + 6.0 * flags["has_date"]

    eff_bits = max(0.0, base_bits - penalties)

    guesses = 2.0 ** eff_bits if eff_bits < 60 else float("inf")
    online_rate = 1e4
    offline_rate = 1e10
    def sec_to_str(sec):
        if sec == float("inf"): return "∞"
        for unit, div in [("s",60),("min",60),("h",24),("d",365),("y",100)]:
            if sec < div: return f"{sec:,.1f}{unit}"
            sec /= div
        return f"{sec:,.1f} centuries"
    t_online  = sec_to_str(guesses / online_rate)  if guesses != float("inf") else "∞"
    t_offline = sec_to_str(guesses / offline_rate) if guesses != float("inf") else "∞"

    feat_map = {name: float(x[0][i]) for i, name in enumerate(_META.get("feature_names", FEATURE_NAMES))}
    return {
        "label": label,
        "proba": proba,
        "features": feat_map,
        "eff_entropy_bits": eff_bits,
        "est_crack_time_online": t_online,
        "est_crack_time_offline": t_offline
    }
