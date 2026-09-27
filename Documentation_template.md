# Business Entity Resolution — Methodology Write-Up
**Amazon ML Challenge 2026 · Team Runtime Rebels**

Final submission: **v4**, leaderboard entity-level macro F0.5 **0.915**
(held-out validation 0.939). Code: `src/assign_pipeline.py` (+ `src/preprocessing.py`,
`src/features.py`).

---

## 1. Executive Summary & Problem Formulation

Source1 is the deduplicated reference set; each Source1 (S1) entity has zero, one or
several true matches in Source2/Source3. The score is **entity-level macro F0.5**:
F0.5 per S1 entity between its predicted and true match sets, averaged over all S1
entities (an entity with no true matches scores 1 only for an empty prediction).

The decisive observation, from the training ground truth: **every S2/S3 record belongs
to at most one S1 entity** (0 of the 7.64M matched records has two owners) and ~27% of
S2/S3 records have no owner at all. So instead of scoring S1 → candidate pairs
independently, the final pipeline is **target-centric**: for every S2/S3 record it asks
*"which single S1 entity owns this record, if any?"* This turned a pairwise
classification problem into a per-record assignment problem and was the largest single
gain (0.744 → 0.888 on the leaderboard).

| Version | Change | Validation | Leaderboard |
|---|---|---|---|
| v1 | rule blocking → pairwise GBDT → dual threshold | 0.745 | 0.744 |
| v2 | target-centric TF-IDF retrieval + assignment; anyascii transliteration | 0.914 | 0.888 |
| v3 | candidate-relative features, LightGBM + CatBoost blend | 0.922 | 0.895 |
| **v4** | second (name) retrieval channel, phonetic skeleton features, zero-padding fix | **0.939** | **0.915** |

---

## 2. Candidate Generation (Retrieval)

Per country (retrieval never crosses countries), `retrieve()` runs two TF-IDF channels
with sparse top-k matrix products (`sparse_dot_topn`):

1. **Combined channel** — word TF-IDF over `name_clean + address_clean`
   (sublinear tf, tokens in more than 5,000 S1 records dropped); top **10** S1 records
   per S2/S3 record by cosine.
2. **Name channel** — character 3–5-gram (`char_wb`) TF-IDF over `name_clean` alone
   (min_df 2, max_df 1% of S1); top **10**. It catches owners whose address text
   diluted the combined cosine.

The two lists are unioned (~18.8 candidates per record, 194M training pairs) and every
candidate receives **both** cosines (the channel that did not retrieve it is computed
exactly for that pair).

**Measured owner recall** (share of matched S2/S3 records whose true owner is among
their candidates, full training set): combined channel alone 92.8%, name channel alone
63.1%, **union 95.3%** (v1 rule-based blocking reached ~70% with ~200 candidates per S1).
With a perfect classifier this retrieval caps validation macro F0.5 at 0.981.

Retrieval-miss analysis drove two preprocessing fixes (`src/preprocessing.py`):
- **Transliteration** with `anyascii` instead of NFKD + ASCII-drop, which had deleted
  Devanagari/Bengali/Telugu/Tamil/Odia/Kannada names outright.
- **Zero-padded numbers** normalised (`House No-008` → `8`, `A-0060` → `60`).

Widening further was measured and rejected: 20 + 20 candidates reach only 96.0% recall
at twice the pairs; the residual misses are generic names ("Smart Enterprises") whose
counterpart has a heavily truncated address.

---

## 3. Feature Engineering

44 features per (S2/S3 record, S1 candidate), `feature_matrix()`:

- **Retrieval (15):** combined cosine, its rank / gap to the record's best / margin to
  the next candidate, number of candidates; how many records ranked this S1 first or
  retrieved it at all, where this record ranks among all records that retrieved the
  same S1 (`s1_reverse_rank`) and its cosine gap to that S1's best record; name-channel
  cosine, rank and gap; which channel(s) retrieved the pair (`both_channels_retrieved`).
- **String similarity (20, `src/features.py::compute_pair_features`):** RapidFuzz
  token-sort/set, partial, ratio, WRatio, Jaro-Winkler on names; token, partial,
  Jaro-Winkler, LCS on addresses; country/first-word/prefix match, length differences,
  numeric-token overlap, token Jaccard, name 2-gram Jaccard / 3-gram Dice, postal-code
  match. Computed in a fork process pool over 1M-pair chunks.
- **Phonetic skeleton (2):** RapidFuzz token-set ratio and ratio between *consonant
  skeletons* of the two names — aspirates folded, soft c/g folded, similar consonants
  merged (t/d, p/f, k/g/c, s/z/j, m/n, b/v/w), vowels dropped, transliterated legal
  forms removed. An English name and its Indian-script transliteration then collide:
  "Southern Projects" / "सदर्न प्रोजेक्ट्स" → `strn prskts`; "Sky Trading" /
  "स्काई ट्रेडिंग" → `sk trtnk`.
- **Candidate-relative (7):** for the key similarities, the gap to the best value among
  the same record's candidates. Assignment picks one owner among ~19 candidates, so how
  a candidate compares with its competitors matters as much as its absolute similarity.
  `skel_token_set_ratio_gap_to_best` is the single most important v4 feature (22% of
  LightGBM gain).

---

## 4. Model Architecture & Training

- **Split:** 20% of S1 entities held out for validation (entity-level, so no entity's
  records leak across the split); training uses a 30% sample of S2/S3 records with all
  of their candidates (46.6M pairs, 3.8% positive), so candidate-relative features
  are exact.
- **Models:** LightGBM (255 leaves, lr 0.05, early stopping → 221 trees) and CatBoost
  (depth 8, lr 0.15, 1,500 trees), both early-stopped on a held-out slice of
  validation-entity pairs.
- **Blend:** probabilities blended as `w·LightGBM + (1−w)·CatBoost`, with `w` chosen on
  held-out entities by the real metric: w = 1.00 → 0.9363, 0.75 → 0.9376,
  0.50 → 0.9386, **0.25 → 0.9390**, 0.00 → 0.9390.
- **Memory:** features for the non-sampled 136M pairs are computed and scored in
  target-aligned 15M-row chunks, so the full feature matrix is never materialised
  (peak 57 GB on a 61 GB instance).

---

## 5. Assignment & Threshold

Each S2/S3 record is assigned to its **single highest-probability S1 candidate** if that
probability clears a threshold, otherwise to nobody. This enforces the one-owner
structure exactly and makes singleton protection implicit: an S1 entity is predicted
empty unless some record chooses it.

The threshold is grid-searched (0.05–0.95, step 0.01) directly against entity-level
macro F0.5 on the held-out S1 entities, using a vectorised implementation of the metric:

```
F0.5_i = 1                              if T_i = P_i = ∅
       = 0                              if exactly one of T_i, P_i is empty
       = 1.25·Prec_i·Rec_i / (0.25·Prec_i + Rec_i)   otherwise
Score  = mean over all S1 entities
```

Chosen: **0.63** (India 0.61, US 0.63 when tuned per country — each country's entities
depend only on that country's assignments, so per-country tuning is exact; it changed
the score by < 0.001). Validation by country: US 0.949, India 0.924.

---

## 6. Zero-Shot Generalization (France)

Training contains only US and India; the test set adds France (15% of test S1). Nothing
branches on country except that retrieval and TF-IDF vocabularies are built per
country, which adapts automatically to French vocabulary and word frequencies.
`anyascii` folds accents (`Société` → `societe`), French legal forms (SARL, SAS, SASU,
EURL, SCI, SNC, GIE, …) are stripped, and French street abbreviations (`bd`, `chem`) are
expanded. France uses the global threshold. Its test behaviour is in line with the
labelled countries (in the v2 output, 5.1% of French S1 entities were predicted empty
vs 5.6% empty in the training ground truth). The consistent ~0.025 gap between validation and leaderboard
across v2–v4 is attributed to France (no labels) and the test set's heavier India
share.

---

## 7. Academic Integrity & Fair Play Statement

No external APIs, geocoding, internet lookups or external data. All signal comes from
the three provided files (name, address, country) via open-source libraries (pandas,
numpy, scikit-learn, sparse_dot_topn, RapidFuzz, anyascii, LightGBM, CatBoost; MIT/BSD/
Apache-2.0). No pretrained or neural models. We checked the data for construction
artefacts (row order, ID numbering vs owner) and found none; no identifier-based
shortcuts are used. Every step is deterministic given the seed and reproducible from
`requirements.txt` and `src/`.

---

## 8. Known Limitations / Next Steps

- **Retrieval ceiling:** 95.3% owner recall caps validation F0.5 at 0.981; wider
  character/word channels plateau near 96%. A retrieval model trained on the ground
  truth (e.g. learned token weights) is the most promising next step.
- **Classifier gap:** 0.939 vs the 0.981 ceiling; more training data was limited by the
  61 GB instance (features for 30% of records).
- **France** cannot be validated without labels.
- Training + test prediction take ~3.7 h on an 8-vCPU / 61 GB EC2 instance.
