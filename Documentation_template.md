# Business Entity Resolution — Methodology Write-Up
**Amazon ML Challenge 2026**

---

## 1. Executive Summary & Problem Formulation

The task is entity resolution across three heterogeneous business-record sources.
Source1 is the deduplicated reference set; each Source1 entity has **zero** (a
singleton), **one**, or **multiple** true matches in Source2/Source3. Predictions are
scored with **entity-level macro F0.5** — F0.5 is computed independently for every
Source1 entity by comparing its predicted match set against its true match set, then
averaged across all entities (not a row-level metric over flattened pairs — see
Section 5). Because a singleton scores 1.0 for an empty prediction and 0.0 for even
one false positive, precision on low-confidence entities matters far more than raw
recall, which shapes the entire downstream design: high-recall blocking, precision-
weighted features, and a threshold rule built specifically to protect singletons.

The pipeline is implemented as a modular `src/` package: `preprocessing.py` →
`blocking.py` → `features.py` → `training_data.py` → `train.py` →
`threshold_search.py` → `inference.py`, each independently testable (60 unit tests)
and independently runnable via CLI.

---

## 2. Candidate Generation (Blocking) Strategy

Naive matching is O(|Source1| × |Source2 ∪ Source3|), which is intractable at this
dataset's scale. `src/blocking.py` instead builds a multi-index inverted-blocking
pipeline: every record is hashed under several cheap keys, and only records sharing
at least one key are ever compared.

**Rules** (each an independent recall channel, unioned and deduplicated):
1. Country + first 3 characters of the cleaned name
2. Country + first word of the cleaned name
3. Country + prefix of the cleaned address
4. Token-based inverted index over name tokens (rare-token filtered)
5. Numeric-token overlap on addresses (house/unit numbers)
6. Country + last word of the cleaned name (catches prefix drift, e.g. "St"/"Saint")
7. Country + sorted token-initials acronym (catches word reordering)
8. Country + Metaphone phonetic code of the first name token (catches transliteration
   variants, e.g. "Kumar" vs. "Coomar" — common in Indian business names)

**Capped document frequency** (`BROAD_KEY_MAX_DF_RATIO = 0.01`, plus per-rule
`max_df_ratio` on the name-token rule) drops any block key shared by more than 1% of
rows before pair generation. This is not a minor tuning knob: at full dataset scale a
single common key (e.g. a generic first word) forms thousands of individually
medium-sized blocks that each pass a per-block size cap yet sum to tens of millions of
low-value pairs — confirmed directly during a full-scale run, where one unguarded rule
alone produced 14.2M raw pairs against Source2. Capping frequency before pair
generation, combined with a hard per-block cross-product cap (`max_block_pairs`), is
what keeps both memory and candidate volume tractable.

**Measured on held-out validation samples:** recall ≈ 96.4–96.7%, reduction ratio
≈ 99.3–99.6% versus the full cross product. (Full-dataset-scale numbers are pending a
higher-memory environment than local development allowed — see Section 8.)

---

## 3. Feature Engineering

`src/features.py` computes **23 features** per candidate pair from `name_clean` /
`address_clean` (all RapidFuzz scoring is C-accelerated; no `iterrows()` anywhere):

- **Name similarity (6):** `token_sort_ratio`, `token_set_ratio`, `partial_ratio`,
  `ratio`, `WRatio`, Jaro-Winkler
- **Address similarity (4):** token-sort ratio, partial ratio, Jaro-Winkler,
  longest-common-subsequence ratio
- **Structural (6):** exact country match, first-word match, prefix match, name/address
  length difference, numeric-token Jaccard overlap
- **Character n-gram (2):** 2-gram Jaccard, 3-gram Dice coefficient on names
- **Address-specific (1):** postal/PIN code exact match (5–6 digit trailing numeric
  token — country-agnostic by construction, never branches on country)
- **Group-relative ranking (3):** `rank_within_entity`, `is_top1_for_entity`,
  `score_margin_to_next` — a candidate's standing *among all candidates blocked for
  the same Source1 entity*, computed via vectorized `groupby` rank/transform. This
  lets the model use competitive context (e.g. "clearly the best of 40 candidates" vs.
  "similar in isolation but barely ahead of a rival") that pairwise features alone
  cannot express.

---

## 4. Model Architecture & Training

`src/training_data.py` builds a **6:1 hard-negative-to-positive** labeled set:
positives from ground truth, negatives sampled from candidate pairs that passed
blocking but are not true matches (far more informative than random pairs, since
blocking would never propose a genuinely dissimilar pair). The 1:1 ratio tried
earlier taught the model to over-predict matches; 6:1 better reflects the true
class imbalance without starving the positive class at this dataset's scale.

The train/validation split is **entity-level** (never split by individual pair, to
avoid leakage) and **stratified** by (match-count bucket: singleton / single-match /
multi-match) × country, so local CV mirrors the true Source1 population rather than
an arbitrary random draw — important with only 5 submissions/day.

`src/train.py` trains four candidates: **Logistic Regression** (standardized,
class-balanced baseline), **LightGBM**, **XGBoost**, and **CatBoost**, each with
early stopping on the validation set, plus a fifth: an **ensemble** that
probability-averages the three GBDT models. All five compete on equal footing for
best-model selection; the ensemble does not automatically win (observed: it slightly
underperformed solo CatBoost in one run), so selection is always validated against
the real entity-level metric (Section 5), never the diagnostic alone.

---

## 5. The Dual-Threshold Strategy

**Entity-level Macro F0.5** (the actual competition metric — `src/metrics.py`):

```
For Source1 entity i with true match set T_i and predicted match set P_i:

  F0.5(P_i, T_i) = 1.0                                     if T_i = P_i = empty
                 = 0.0                                     if exactly one of T_i, P_i is empty
                 = 1.25 * Prec_i * Rec_i / (0.25*Prec_i + Rec_i)   otherwise

  where Prec_i = |T_i ∩ P_i| / |P_i|,  Rec_i = |T_i ∩ P_i| / |T_i|

  Score = (1 / |Source1|) * sum_i F0.5(P_i, T_i)
```

This is deliberately **not** `sklearn.fbeta_score(average="macro")` run on flattened
pairs — that averages over the match/non-match *classes*, a different number
entirely. Confusing the two was an early bug in this project; fixing it changed the
measured score materially (a pair-level proxy showed ~0.997 on a validation slice,
while the correct entity-level metric on the same model showed ~0.94–0.96).

**Decision rule**, for each Source1 entity's candidate probabilities {p₁, ..., pₖ}:
- If `max(p) < T_singleton`: predict the empty set (protects the 1.0 singleton score)
- Else: accept every candidate with `p ≥ T_match` (`T_match ≥ T_singleton`)

`src/threshold_search.py::search_dual_thresholds` grid-searches `(T_singleton,
T_match)` directly against entity-level macro F0.5 on the **full candidate pool**
(never the class-balanced training sample, whose precision numbers do not reflect the
real, heavily-imbalanced candidate distribution), using a precomputed sorted/bisect
structure per entity for speed. `src/inference.py` applies the resulting thresholds
and writes `matching_results.tsv` (subset of) `candidate_pairs.tsv`, both validated by
`utils/validate_submission.py` before packaging.

---

## 6. Zero-Shot Out-of-Domain Generalization (France)

Training data contains only US and India; the test set introduces France. Country is
never hardcoded, filtered, or branched on anywhere in the pipeline — generalization
comes entirely from text normalization in `src/preprocessing.py`:

- **Unicode NFKD accent normalization**: `Société` → `societe`, `Café` → `cafe`,
  applied *before* the ASCII-only punctuation regex (which would otherwise silently
  delete accented characters instead of folding them).
- **French legal-entity suffixes** added alongside the original set: SARL, SAS,
  SASU, SA, EURL, SCI, SNC, GIE, CIE, Association, GmbH.
- **Street-designator synonyms** (`rd`→road, `ave`→avenue, `blvd`/`bd`→boulevard,
  `rte`→route, `chem`→chemin) so equivalent addresses block/match regardless of
  abbreviation — deliberately excluding "st" (Street vs. Saint is genuinely
  ambiguous; guessing wrong would corrupt real content).

Verified against real French rows from `test_source1.tsv` (e.g. `ZNB Club SARL` →
`znb club`, `Maison de Santé Generation` → `maison de sante generation`).

---

## 7. Academic Integrity & Fair Play Statement

This solution uses **no external API calls, geocoding services, internet lookups, or
external databases** of any kind. All matching signal is derived exclusively from the
three provided source files (business name, address, country) using open-source
libraries (pandas, RapidFuzz, jellyfish, scikit-learn, LightGBM, XGBoost, CatBoost),
all MIT/BSD/Apache-2.0 licensed, none exceeding the parameter-count limit (the GBDT
models are not parameter-counted neural networks; no pretrained model of any kind is
loaded from the network). Every transformation is deterministic and reproducible from
`requirements.txt` and the `src/` pipeline alone.

---

## 8. Known Limitations / Next Steps

Local development happened on a memory-constrained machine (~1–2GB free RAM), which
bounded end-to-end validation to representative held-out samples rather than the full
2.2M/5M/5.3M-row dataset. The `dtype=str` fix in `src/data_loader.py` and the
`max_df_ratio` blocking fix in `src/blocking.py` were specifically added to make a
full-scale run tractable; both are unit-tested and validated on samples, with a
full-scale confirmation pending a higher-memory environment (e.g. the AWS EC2 instance
provisioned via `setup_ec2.sh`).
