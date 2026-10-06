# Modeling decisions

## Why LDA first

Linear Discriminant Analysis is the real-world clinical standard for
myoelectric control, fast enough for real-time use in actual prosthetic
devices, not just a convenient scikit-learn default. Using it as the
first baseline is a domain-authentic choice, not just "the easy one to
try first."

## Feature columns: 40, not 50

ZC columns are excluded here, at modeling time, not deleted from the
saved `.npz` files. See FEATURES.md for the full reasoning: the signal
is confirmed rectified, so ZC is structurally constant (zero variance),
which is a real numerical-stability risk for LDA's covariance-based fit,
not just an uninformative column. `src.baselines.select_feature_columns`
does this exclusion, tested to confirm it can not silently exclude
everything (a guard against a typo'd prefix removing the whole feature
set without any error).

## Standardization

`StandardScaler`, fit on train only, applied to train and validation.
LDA's `svd` solver (the default, used here explicitly) does not strictly
require scaling the way distance-based methods do, but the five features
here span genuinely different scales and types, continuous amplitude
measures (MAV, RMS, WL) alongside SSC, a small integer count. And
without scaling, a feature could dominate the covariance structure by
raw magnitude rather than genuine discriminative power. Fitting the
scaler on train only (never on validation) avoids leaking validation
statistics into the model, the same discipline as fitting the label
encoding and every other train-derived artifact only on train.

## Evaluation design: validation set + subject-grouped CV, not either alone

Two numbers, answering two different questions:

1. **Validation accuracy** (3 subjects, never trained on): the actual
   deployment question, does this generalize to a genuinely new person.
2. **6-fold GroupKFold CV within the 18 train subjects** (3 held-out
   subjects per fold, matching the validation set's size on purpose, so
   the two numbers are directly comparable): a robustness check on that
   single validation estimate. With only 3 validation subjects, one
   estimate could be an unusually easy or hard draw. The real
   per-subject heterogeneity has already been found in this dataset
   (anomaly rates from 6.40% to 12.94% across just four 3-subject groups),
   so this is not a hypothetical concern.

`train_lda_baseline.py` flags automatically if validation accuracy falls
outside the CV fold range by more than 2 standard deviations from the CV
mean, worth investigating if it fires, not something to explain away.

GroupKFold ensures no subject appears on both sides of any fold,
tested directly in `tests/test_baselines.py`, not just assumed from
GroupKFold's documentation.

## Per-subject accuracy, not just an aggregate

`src.baselines.per_subject_accuracy` is reported for the validation set
and reused wherever a later baseline needs the same breakdown
rather than rebuilt each time. An aggregate accuracy can hide a
model that works well for most subjects and badly for one or two,
worth knowing which, given the per-subject variation already documented
throughout this project.

## What gets saved

- `models/lda_baseline.joblib`: the fitted model, the fitted scaler, and
  the exact 40 feature column names used, bundled together so the
  model can never accidentally be applied to a differently-ordered or
  differently-sized feature vector later without an explicit mismatch.
- `reports/lda_baseline_results.json`: validation accuracy/macro-F1,
  per-subject breakdown, CV fold scores and summary stats, and which
  columns were used/excluded, a structured, versioned record, not just
  console output that disappears at the end of the terminal session.

## Results: the accuracy/macro-F1 split is real, and priors do not fix it

First real run, empirical (default) priors: validation accuracy 0.6260,
macro-F1 0.1099. Accuracy driven almost entirely by rest (~59% of the
data), active-class performance is actually poor. The CV-vs-validation
consistency check did not fire (CV mean accuracy 0.6014, CV mean
macro-F1 0.1003. Both close to the validation numbers, so this is not
an unusually easy or hard validation draw).

Hypothesis 1 tested: majority-class bias from the empirical prior.
Re-run with `priors` forced uniform: accuracy fell to 0.4051 (losing on
rest, which empirical priors were getting mostly right) while macro-F1
only rose to 0.1202. A real but marginal move, not a fix, most of the
macro-F1 ceiling survives the prior change untouched.

Hypothesis 2 tested: LDA's shared-covariance assumption. Built
`scripts/train_qda_diagnostic.py` to isolate exactly this one variable.
QDA fits a separate covariance matrix per class instead of one shared
matrix, with every other part of the pipeline (features, scaler
discipline, split, CV protocol, priors comparison) held identical to
the LDA run. Result:

| Setting           | LDA macro-F1 | QDA macro-F1 |
|--------------------|-------------:|-------------:|
| Empirical priors   | 0.1099       | 0.1083       |
| Uniform priors      | 0.1202       | 0.1076       |

QDA does not meaningfully beat LDA under either prior - if anything it is
slightly worse, and validation accuracy is also slightly lower under
QDA (0.5869 vs 0.6260, empirical). **Shared covariance was not the
problem.** Both hypotheses tested so far (priors, covariance structure)
leave the macro-F1 ceiling essentially unmoved, which points at the
features themselves (MAV/RMS/WL/SSC) not carrying enough
linearly-or-quadratically-usable signal to separate most of the 40
active classes, consistent with the identified overlap cluster
(`2_1, 2_8, 2_9, 2_11, 2_12, 3_1, 3_5, 3_6`). This makes CNN
(learning its own features from the raw signal) load-bearing for this
project, not just a comparison baseline.

One data-quality finding surfaced by the QDA run specifically: fitting
raised "covariance matrix ... is not full rank" warnings for 25 of the
41 classes (161 warnings across the 6 CV folds, consistently, roughly
25-27 per fold). This is not a thin-class problem (the smallest
training class has 10,560 samples against 40 features, far above the
`5x` rule-of-thumb margin checked up front). It's consistent with the
MAV/RMS near-redundancy finding: two near-duplicate feature blocks
make any per-class covariance matrix structurally close to singular
regardless of sample size. Worth a closer look before relying on any
covariance-based method again (dropping one of MAV/RMS, or combining
them), not addressed here since it doesn't change this diagnostic's
conclusion either way (QDA still does not help even in the one feature
block that is conditioned fine).

Per-subject pattern holds across every variant tried (LDA x2 priors,
QDA x2 priors): subject 23 is consistently the weakest of the three
validation subjects (e.g. 0.580 under LDA-empirical, 0.527 under
QDA-empirical) while 16 is consistently the strongest. Not acted on
now, but worth carrying into error analysis, a subject-level
effect this consistent across four different model fits is unlikely to
be noise.

## Baseline model decision, reversed

The uniform-priors model was originally saved as `models/lda_baseline.
joblib` ("the baseline going forward"), on the reasoning that it pushed
macro-F1 up. That decision is reversed: the QDA diagnostic shows the
priors were never the actual fix for the macro-F1 ceiling (QDA's
macro-F1 is flat across priors too, under a completely different
decision rule), so uniform priors was paying a real ~22-point accuracy
cost for a ~1-point macro-F1 gain that does not track back to anything
that was actually broken. Empirical priors is also the more honest fit
for how this would run in practice, rest genuinely is ~59% of real
myoelectric signal, so a model that does not know that is less
representative of deployment, not more. `scripts/train_lda_baseline.py`
now saves the **empirical-priors** model as the baseline
(`saved_model_priors: "empirical"` in the results JSON). Macro-F1
keeps being reported as the headline diagnostic metric regardless of
which model is saved. This reversal is about which model file to keep
around, not about which number matters.

## Next: LinearSVC, reframed

A second classical baseline (LinearSVC - a linear,
margin-based, discriminative method, as opposed to LDA/QDA's
generative, Gaussian-likelihood approach). With two hypotheses already
ruled out (priors, covariance structure), LinearSVC becomes a
confirmatory third data point rather than just "another model to try":
if it also caps out around macro-F1 ~0.10-0.12, that's three
structurally different classical approaches hitting the same wall,
which is about as strong a case as classical ML can make for "the
limitation is the features, not the model family" before moving to
the CNN.