Lead Scoring Case

::: From predictive signals to real decision drivers ::: 

Not every strong predictor is a real lever.
This case separates what predicts conversion from what actually explains and sustains it.
Executive Focus
Most lead-scoring systems optimize for AUC.
This case optimizes for decision reliability.

We distinguish between:

🔮 Predictive performance (what improves model accuracy)
🧭 Structural truth (what remains stable, causal-plausible and decision-relevant)

The goal is not to win a modeling contest.
The goal is to identify robust levers worth acting on.


::: Core Questions :::

1. Which features explain conversion in a stable and reproducible way?
2. Which effects disappear when tested across resamples and model setups?
3. Which signals are overestimated by predictive models?
4. Which features are safe to scale — and which should be stopped?


::: Why This Matters ::: 

In many real-world lead scoring systems:
    Dozens of features look “significant”
    Dashboards highlight short-term effects
    Teams act on unstable signals

This leads to:
    Budget waste
    Tactical over-optimization
    Endless discussions without structural clarity

This project takes a different approach:
    Robustness over complexity.
    Decisions over metrics.



::: Methodological Framework ::: 


The pipeline is built around three decision gates.


1️⃣ Structural vs Predictive Modeling

Two models are trained:
    Predictive Logit
    Maximizes performance (AUC, recall, precision)
    Structural Logit
    Removes process-proximity features and leakage
    Focuses on interpretable, decision-relevant signals

This dual setup allows detection of:
    Overestimated features (predictive strong, structural weak)
    Underestimated features (structural strong, predictive ignored)


2️⃣ Statistical Evidence (Gate 2)

Effects are validated using:
    Bootstrap confidence intervals
    Multiple testing correction (BH-FDR)
    q-value thresholding

Only effects passing statistical correction are considered further.


3️⃣ Robustness & Stability (Gate 3)

Features must also satisfy:
    Minimum effect size (|OR − 1|)
    Sign stability across bootstrap samples
    Decision consistency across setups

Each feature is classified as:
    SCALE
    INVESTIGATE
    STOP



::: What Makes This Different ::: 

This is not:
    A Kaggle-style leaderboard exercise
    A feature importance ranking
    A dashboard project

This is:
    A robustness-first analysis
    A structural interpretation framework
    A decision-oriented pipeline

The output is not “interesting coefficients”.
The output is actionable prioritization.



::: Data ::: 

Public Kaggle Lead Scoring dataset
Raw data stored locally (not committed)
Repository contains:
    Full analysis pipeline
    Reproducible configuration
    Automated reporting

See data/README.md for reproduction details.



::: Automated Outputs ::: 

Running:
    python -m src.case

Generates:
    Structural vs Predictive comparison tables
    Overestimated & underestimated feature reports
    Segment profiles
    Decision evidence tables
    Executive summary (Markdown)
    Full structured report
    Exportable lead lists

All artifacts are written to:
    results/



::: Philosophy ::: 

A feature is only valuable if it is:
    Statistically supported
    Stable across resamples
    Structurally interpretable
    Actionable in real operations
Everything else is noise —
no matter how good it looks in a single model.



::: Final Takeaway ::: 

Predictive power is easy.
Decision reliability is hard.

This case demonstrates how to build a lead scoring system
that withstands validation, scrutiny and operational reality.