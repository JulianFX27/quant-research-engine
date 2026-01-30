# Dataset Contract — Backtester (v1.0)

This document defines the **formal dataset contract** for the Backtester / Quant Research Engine.
It is normative. Code must conform to this contract.

---

## 1. Dataset Identity

Each dataset has **two identities**:

### 1.1 Semantic Identity (`dataset_id`)
A human-readable identifier encoding dataset meaning.

**Format:**
<INSTRUMENT><TIMEFRAME><START_DATE><END_DATE><SOURCE>

**Example:**
EURUSD_M5_2026-01-01__2026-01-02__csv_example


**Rules:**
- `dataset_id` represents *meaning*, not content.
- Any semantic change **requires a new dataset_id**, including:
  - different date range
  - different source
  - filtering (NY-only, clean_v2, etc.)
  - different instrument or timeframe

---

### 1.2 Canonical Identity (`fingerprint_sha256`)
A cryptographic hash of the **canonicalized dataset content**.

**Properties:**
- Deterministic
- Content-based
- Independent of row order
- Sensitive to any OHLC or timestamp change

**Rule:**
> One `dataset_id` MUST map to exactly one `fingerprint_sha256`.

---

## 2. Dataset Fingerprint (Canonicalization)

The fingerprint is computed from:

- Timestamps (normalized to UTC, ns)
- OHLC values (float64)
- Sorted by timestamp
- Duplicate timestamps are forbidden

**Consequences:**
- Any content mutation produces a different fingerprint.
- Fingerprint is the ultimate guard against silent data corruption.

---

## 3. Schema Identity (`schema_sha256`)

A secondary hash representing **structural compatibility**.

Includes:
- time column (or index sentinel)
- OHLC column names
- dtypes
- fingerprint version

**Scope:**
- Only columns consumed by the engine are part of the schema.
- Auxiliary columns do not affect schema unless explicitly included.

---

## 4. Dataset Registry

The registry enforces the identity contract.

### 4.1 Registry Files
data/registry/
├── datasets.jsonl # append-only audit log
└── datasets_latest.json # latest snapshot by dataset_id

Registry is **local runtime-only** and is NOT versioned in git.

---

### 4.2 Registry Policy

| Situation | Action |
|---------|--------|
| New `dataset_id` | REGISTER |
| Existing `dataset_id` + same fingerprint | MATCH (OK) |
| Existing `dataset_id` + different fingerprint | ERROR |
| Override requested | Allowed only if explicit + audited |

---

### 4.3 Error Codes (Stable)

- `DATASET_ID_FINGERPRINT_MISMATCH`
- `DATASET_ID_PROVISIONAL_NOT_ALLOWED`
- `DATASET_ID_OVERRIDE_REASON_REQUIRED`

Error strings are part of the public contract and MUST NOT change.

---

## 5. Overrides (Exceptional)

Overrides are **explicit and audited**.

Requirements:
- `allow_override = true`
- `override_reason` MUST be non-empty

Override events:
- Are logged in `datasets.jsonl`
- Update `datasets_latest.json`
- Are considered exceptional actions

---

## 6. Provisional Dataset IDs

Provisional IDs (e.g. `unknown__unknown`) are **forbidden** in the registry.

Rule:
> Dataset IDs must be finalized using canonical start/end timestamps BEFORE registry interaction.

---

## 7. Reproducibility Contract

Every run MUST persist full dataset metadata in:

results/runs/<RUN_ID>/run_manifest.json

Mandatory fields:
- `dataset_id`
- `fingerprint_sha256`
- `schema_sha256`
- `fingerprint_version`
- provenance (file hash, path if available)

**Reproducibility Rule:**
> A run is reproducible if and only if dataset fingerprint + config hash match.

---

## 8. Smoke Validation

The following MUST pass:

make smoke


Smoke tests enforce:
- deterministic fingerprinting
- registry correctness
- mutation detection
- manifest dataset persistence

---

## 9. Stability Rules

- Datasets used in research are immutable.
- Historical results are never recomputed with modified data.
- New data ⇒ new dataset_id.
- Infra changes require explicit versioning.

---

**End of Contract**
