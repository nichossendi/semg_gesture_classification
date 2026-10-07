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
measures (MAV, RMS, WL) alongside SSC, a small integer count and
without scaling, a feature could dominate the covariance structure by
raw magnitude rather than genuine discriminative power. Fitting the
scaler on train only (never on validation) avoids leaking validation
statistics into the model, the same discipline as fitting the label
encoding and every other train-derived artifact only on train.

## Evaluation design: validation set + subject-grouped CV, not either alone

Two numbers, answering two different questions:

1. **Validation accuracy** (3 subjects, never trained on): the actual
   deployment question, does this generalize to a genuinely new person?
2. **6-fold GroupKFold CV within the 18 train subjects** (3 held-out
   subjects per fold, matching the validation set's size on purpose, so
   the two numbers are directly comparable): a robustness check on that
   single validation estimate. With only 3 validation subjects, one
   estimate could be an unusually easy or hard draw, real per-subject
   heterogeneity has already been found in this dataset (anomaly rates
   from 6.40% to 12.94% across just four 3-subject groups), so this
   is not a hypothetical concern.

`train_lda_baseline.py` flags automatically if validation accuracy falls
outside the CV fold range by more than 2 standard deviations from the CV
mean, worth investigating if it fires, not something to explain away.

GroupKFold ensures no subject appears on both sides of any fold,
tested directly in `tests/test_baselines.py`, not just assumed from
GroupKFold's documentation.

## Per-subject accuracy, not just an aggregate

`src.baselines.per_subject_accuracy` is reported for the validation set
and reused wherever a later baseline needs the same breakdown,
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
macro-F1 0.1099, accuracy driven almost entirely by rest (~59% of the
data), active-class performance is actually poor. The CV-vs-validation
consistency check did not fire (CV mean accuracy 0.6014, CV mean
macro-F1 0.1003, both close to the validation numbers, so this is not
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

QDA does not meaningfully beat LDA under either prior, if anything it is
slightly worse, and validation accuracy is also slightly lower under
QDA (0.5869 vs 0.6260, empirical). **Shared covariance was not the
problem.** Both hypotheses tested so far (priors, covariance structure)
leave the macro-F1 ceiling essentially unmoved, which points at the
features themselves (MAV/RMS/WL/SSC) not carrying enough
linearly-or-quadratically-usable signal to separate most of the 40
active classes, consistent with previous identified overlap cluster
(`2_1, 2_8, 2_9, 2_11, 2_12, 3_1, 3_5, 3_6`). This makes the CNN
(learning its own features from the raw signal) load-bearing for this
project, not just a comparison baseline.

One data-quality finding surfaced by the QDA run specifically: fitting
raised "covariance matrix ... is not full rank" warnings for 25 of the
41 classes (161 warnings across the 6 CV folds, consistently, roughly
25-27 per fold). This is not a thin-class problem (the smallest
training class has 10,560 samples against 40 features, far above the
`5x` rule-of-thumb margin checked up front). It is consistent with the
previous MAV/RMS near-redundancy finding: two near-duplicate feature blocks
make any per-class covariance matrix structurally close to singular
regardless of sample size. Worth a closer look before relying on any
covariance-based method again (dropping one of MAV/RMS, or combining
them), not addressed here since it does not change this diagnostic's
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

## Day 7: LinearSVC, a discriminative counterpoint

LDA and QDA are both generative, Gaussian-likelihood methods: each models
a per-class feature distribution and classifies by which distribution a
point more likely came from. Both hit the same macro-F1 ceiling under
every prior/covariance variant tried. `scripts/train_linearsvc_baseline.py`
tests whether that is a property of the *generative* approach itself, not
just priors or covariance structure, by swapping in a discriminative
method that never estimates a distribution or inverts a covariance
matrix. LinearSVC just fits a max-margin linear boundary directly.
Everything else held identical: same 40 non-ZC columns, same
scaler-fit-on-train discipline, same split, same 6-fold subject-grouped
CV protocol.

Implementation notes, both driven by the dataset's scale
(~1.26M training windows, 40 features):
- `dual = False`, scikit-learn's own guidance for n_samples >> n_features,
  true here by four orders of magnitude.
- `max_iter` raised from sklearn's default (1000), and liblinear's
  `ConvergenceWarning` is caught and reported explicitly rather than left
  to print to stderr where it is easy to miss in a long run. A poor
  macro-F1 from an unconverged fit is a different finding from a
  genuinely poor max-margin fit, and the script says which one it is
  rather than letting the two get conflated.

Class-weight comparison (unweighted vs. `class_weight="balanced"`) runs
parallel to the previous priors comparison, same question, asked of a
discriminative method's loss function instead of a generative method's
decision rule. The script applies the same discipline previously learnt the
hard way: it does not default to saving whichever variant has the higher
macro-F1. It only prefers the balanced model if the macro-F1 gain clears
the run's own CV fold-to-fold spread (the noise floor) *and* the accuracy
cost stays under 10 points; otherwise it saves the unweighted model and
says so explicitly, console output included. Verified against two
synthetic scenarios before trusting it on real data: one where balanced
weights genuinely help (clearly separable classes, imbalanced counts -
script correctly saved the balanced model) and one with heavy class
overlap mirroring the real EMG ceiling (balanced and unweighted converged
to the same result, script correctly saved unweighted, matching the
"do not assume the bias-correction is a free win" previous lesson).

Also pulls in `lda_baseline_results.json` and `qda_diagnostic_results.json`
if present (never hardcoded) for a single six-row macro-F1 table across
every generative + discriminative variant tried so far, plus the spread
across all of them. The single number that answers the question this
whole exercise was set up to answer:

  - **Spread stays narrow (< ~0.05) across all variants** -> three
    structurally different classical approaches, shared-covariance
    generative, per-class-covariance generative, margin-based
    discriminative, land in the same place regardless of how
    class-imbalance is handled. That is about as strong a case as
    classical ML can make that the ceiling is the hand-crafted features
    (MAV/RMS/WL/SSC) themselves, not the model family. Makes the CNN
    (learning its own features from the raw signal) load-bearing for
    this project, not just a comparison baseline.
  - **LinearSVC clears the others by a real margin** -> the generative /
    covariance-based framing itself was part of the limitation, not just
    the features, a more optimistic and genuinely different finding
    from what was previously suggested, worth a closer look at *why* before
    moving on (e.g. whether a kernel SVM or other discriminative method
    is worth a further round before the CNN).

## Current result: the narrow-spread case, confirmed

| Variant | Validation macro-F1 |
|---|---:|
| LDA, empirical priors | 0.1099 |
| LDA, uniform priors | 0.1202 |
| QDA, empirical priors | 0.1083 |
| QDA, uniform priors | 0.1076 |
| LinearSVC, unweighted | 0.0959 |
| LinearSVC, balanced | 0.1125 |

Spread across all six: **0.0243** (0.0959-0.1202). No convergence warnings
on either fit (`max_iter = 5000` was enough both times) and no thin-class
issue, so there is no numerical-artifact explanation for the result either,
these are clean fits. Both class-weight variants converged. The script's
own decision rule correctly declined to save the balanced model (macro-F1
gain of 0.0166 did not clear the run's CV fold-to-fold spread of
0.031-0.036, and cost 3.6 accuracy points). The previous lesson held up
automatically on a genuinely different dataset slice, not just the one it
was written against.

LinearSVC's unweighted macro-F1 (0.0959) is in fact the single worst
result of any variant tried across three model families. A purely
discriminative, max-margin method does not do better than either
generative method here, if anything slightly worse before weighting is
accounted for. Combined with LDA and QDA's results, **three structurally
different classical approaches, shared-covariance generative,
per-class-covariance generative, margin-based discriminative, all land
within a quarter-point-of-macro-F1 band (~0.08-0.12), regardless of how
class imbalance is handled in each.** That rules out model family as the
lever and leaves the hand-crafted features (MAV/RMS/WL/SSC) as the
remaining explanation by elimination across three separate, independent
tests, not just one.

**Decision: classical modeling is done for this project's baseline
purposes.** No further classical variant (kernel SVM, random forest,
gradient boosting, further hyperparameter tuning) is planned before the
CNN. The evidence bar for "it is the model, not the features" would need
a result clearly outside this ~0.08-0.12 band, and nothing in the
pattern across six fits from three families suggests more tuning closes
that gap. The CNN, which learns its own features from the raw
signal rather than from MAV/RMS/WL/SSC, is the load-bearing step for
whether this problem is solvable with the available channels at all, not
an optional comparison point.

One practical note for reproducibility: the real run (1.26M training
windows, 41 classes, one-vs-rest under the hood) took ~2.3 minutes per
fit and ~29-30 minutes per 6-fold CV sweep, about 70 minutes total for
both class-weight variants. Worth knowing before kicking off the CNN
training, which will run considerably longer.