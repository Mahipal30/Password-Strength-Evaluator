# trainmodel.py
import os, json, math, re, string
from collections import Counter
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data.csv", on_bad_lines="skip").dropna()
assert {"password","strength"}.issubset(df.columns)

# ---------- Fix class imbalance via downsampling ----------
counts = df["strength"].value_counts()
min_count = counts.min()
print("Original class counts:", {int(k): int(v) for k, v in counts.items()})
balanced_frames = []
for cls in df["strength"].unique():
    subset = df[df["strength"] == cls]
    balanced_frames.append(subset.sample(n=min_count, random_state=42))
df = pd.concat(balanced_frames, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
print("Balanced class counts:", {int(k): int(v) for k, v in df["strength"].value_counts().items()})
print("Total rows after balancing:", len(df))

def corpus_specials(series: pd.Series) -> list:
    sset = set()
    for p in series.astype(str):
        sset.update(ch for ch in p if not ch.isalnum())
    return sorted(sset)
SPECIALS = corpus_specials(df["password"])

def build_ngram(series: pd.Series):
    char_counts = Counter()
    bigram_counts = Counter()
    for p in series.astype(str):
        char_counts.update(p)
        for i in range(len(p)-1):
            bigram_counts[(p[i], p[i+1])] += 1
    vocab = max(1, len(char_counts))
    return {"char_counts": dict(char_counts), "bigram_counts": dict(bigram_counts), "vocab": vocab}

NGRAM = build_ngram(df["password"])

from password_utils import FEATURE_NAMES  

_QWERTY_ROWS = [
    "`1234567890-=","qwertyuiop[]\\","asdfghjkl;\'","zxcvbnm,./"
] + ["`1234567890-="[::-1],"qwertyuiop[]\\"[::-1],"asdfghjkl;\'"[::-1],"zxcvbnm,./"[::-1]]

_LEET_MAP = str.maketrans({"0":"o","1":"l","3":"e","4":"a","5":"s","7":"t","8":"b","9":"g","2":"z","$":"s","@":"a","!":"i"})
_COMMON_WORDS = {
    "password","pass","admin","welcome","letmein","qwerty","iloveyou","monkey",
    "dragon","football","baseball","abc123","login","starwars","princess"
}
def shannon_entropy(s):
    if not s: return 0.0
    L = len(s); freq = [s.count(ch)/L for ch in set(s)]
    return -sum(p*math.log2(p) for p in freq)

def charclass_bits(s, specials_set):
    if not s: return 0.0
    charset = 0
    if any(c.islower() for c in s): charset += 26
    if any(c.isupper() for c in s): charset += 26
    if any(c.isdigit() for c in s): charset += 10
    if any(not c.isalnum() for c in s): charset += max(1, len(specials_set))
    return len(s) * math.log2(max(1, charset))

def ascii_seq_max(s):
    m=1; cur=1
    for i in range(1,len(s)):
        step = ord(s[i]) - ord(s[i-1])
        if step in (1,-1):
            cur += 1; m = max(m, cur)
        else:
            cur = 1
    return m

def qwerty_run_max(s):
    s = s.lower(); best = 1
    for row in _QWERTY_ROWS:
        for k in range(len(s)):
            cur = 1
            for j in range(k+1, len(s)):
                if s[j-1:j+1] in row:
                    cur += 1; best = max(best, cur)
                else:
                    break
    return best

def repeat_longest_run(s):
    longest=1; run=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: run += 1; longest=max(longest,run)
        else: run=1
    return longest

def repeated_sub_fraction(s):
    for w in range(1, len(s)//2 + 1):
        if len(s) % w == 0 and s[:w]*(len(s)//w) == s:
            return 1.0
    return 0.0

def lz_complexity(s):
    seen=set(); w=""; c=0
    for ch in s:
        w+=ch
        if w not in seen:
            seen.add(w); c+=1; w=""
    return c

def avg_bigram_surprisal(s, ngram):
    if not s or len(s) < 2: return 0.0
    cc = ngram["char_counts"]; bc = ngram["bigram_counts"]; vocab=max(1, ngram["vocab"]); k=1
    tot=0.0; n=0
    for i in range(len(s)-1):
        c1, c2 = s[i], s[i+1]
        prob = (bc.get((c1,c2),0)+k) / (cc.get(c1,0)+k*vocab)
        tot += -math.log2(prob); n+=1
    return tot/n if n else 0.0

def pattern_flags(s):
    s_norm = s.translate(_LEET_MAP).lower()
    return {
        "has_year": 1 if re.search(r"(19|20)\d{2}", s) else 0,
        "has_date": 1 if re.search(r"\b(?:\d{2}[-/_]\d{2}[-/_]\d{2,4}|\d{4}[-/_]\d{2}[-/_]\d{2})\b", s) else 0,
        "starts_with_digits": 1 if re.match(r"^\d{2,}", s) else 0,
        "ends_with_digits": 1 if re.search(r"\d{2,}$", s) else 0,
        "has_common_word": 1 if any(w in s_norm for w in _COMMON_WORDS) else 0
    }

def extract_features_train(pw: str, specials_set, ngram):
    s = pw or ""
    L = len(s)
    digits = sum(c.isdigit() for c in s)
    uppers = sum(c.isupper() for c in s)
    lowers = sum(c.islower() for c in s)
    specials = sum((not c.isalnum()) for c in s)

    unique_ratio = len(set(s))/L if L else 0.0
    ratios = (uppers, lowers, digits, specials)
    denom = max(1, L)
    upper_ratio, lower_ratio, digit_ratio, special_ratio = (r/denom for r in ratios)

    flags = pattern_flags(s)
    vec = np.array([
        L, digits, uppers, lowers, specials,
        shannon_entropy(s), charclass_bits(s, specials_set),
        unique_ratio, upper_ratio, lower_ratio, digit_ratio, special_ratio,
        ascii_seq_max(s), qwerty_run_max(s), repeat_longest_run(s), repeated_sub_fraction(s),
        lz_complexity(s), (lz_complexity(s)/denom if L else 0.0), avg_bigram_surprisal(s, ngram),
        flags["has_year"], flags["has_date"], flags["starts_with_digits"], flags["ends_with_digits"], flags["has_common_word"],
        sum([digits>0, uppers>0, lowers>0, specials>0])
    ], dtype=float)
    return vec

X = np.vstack([extract_features_train(p, set(SPECIALS), NGRAM) for p in df["password"].astype(str)])
y = df["strength"].astype(int).to_numpy()

# ---------- Train / Evaluate ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

base_clf = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_leaf=2,
    class_weight="balanced_subsample", n_jobs=-1, random_state=42
)
clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(clf, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(NGRAM, "models/ngram.pkl")
with open("models/meta.json", "w", encoding="utf-8") as f:
    json.dump({"special_chars": SPECIALS, "feature_names": list([
        "length","digits","uppers","lowers","specials",
        "shannon_entropy","charclass_bits",
        "unique_ratio","upper_ratio","lower_ratio","digit_ratio","special_ratio",
        "ascii_seq_max","qwerty_run_max","repeat_longest","repeat_sub_fraction",
        "lz_complexity","lz_norm","avg_bigram_surprisal",
        "has_year","has_date","starts_with_digits","ends_with_digits","has_common_word",
        "variety_classes"
    ])}, f, ensure_ascii=False, indent=2)

print("✅ Saved: models/model.pkl, models/scaler.pkl, models/ngram.pkl, models/meta.json")
