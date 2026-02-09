# Lead Scoring Case

**Which features truly explain conversion — and which are overestimated or just noise?**

This case focuses on separating *stable, decision-relevant signals* from effects that look convincing
but do not hold up across time, segments, or validation splits.

The goal is not to build the most complex model,
but to understand **what actually matters for decisions**.

---

## Core Questions

- Which features explain conversion in a **robust and stable** way?
- Which effects disappear once we test them across time or segments?
- Are observed differences **real effects or statistical noise**?
- Which features are worth acting on — and which should be ignored?

---

## Why This Matters

In many lead-scoring setups:
- dozens of features look relevant
- dashboards highlight point-in-time effects
- decisions are made on unstable signals

This leads to:
- over-optimization
- wasted budget
- endless discussions without clear outcomes

This case takes a different approach:
**robustness over complexity, decisions over metrics.**

---

## Approach

The analysis follows three principles:

### 1. Signal vs. Noise
Effects are tested across:
- multiple data splits
- different time windows
- comparable segments

Only effects that remain stable are treated as meaningful.

### 2. Explanation before Prediction
The focus is on understanding *why* conversion changes,
not only on maximizing predictive accuracy.

### 3. Decision Orientation
Results are translated into:
- clear recommendations
- features to focus on
- features to explicitly ignore

---

## Data

- Public lead scoring dataset (Kaggle)
- Raw data is stored **locally** and not committed to this repository
- The repository focuses on analysis logic and decision-making

See `data/README.md` for details on data handling and reproduction.

---

## Outputs

The repository contains:
- notebooks documenting the analysis logic
- selected figures and tables derived from the analysis
- no raw data

Results are intentionally limited to what supports decisions.

---

## What This Case Is — and Is Not

**This case is:**
- explanatory
- decision-oriented
- focused on stability and robustness

**This case is not:**
- a modeling competition
- a feature dump
- a dashboard exercise

---

## Key Takeaway

A feature is only useful if its effect is:
- real
- stable
- actionable

Everything else is noise — no matter how good it looks in a single chart.