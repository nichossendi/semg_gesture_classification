# Feature engineering decisions

## The rectification question, confirmed on the real data

The raw-trace figures showed both rest and active example segments
staying entirely non-negative, which suggested the signal might already
be rectified. **Confirmed on the real, full 1,890,773-window dataset**:
`check_signal_is_rectified()` returns `True`, no negative value appears
anywhere in `windows_emg.npy`. This is a settled property of this data,
not a contingency to check for anymore.

Consequence: **zero-crossing count (ZC) is structurally degenerate**. A
signal that never goes negative can never cross zero, so ZC returns
exactly 0 for every window, on every channel, regardless of threshold,
confirmed directly, not just reasoned about, since `zc_threshold` cannot
change this outcome (`emg[:, :-1, :] * emg[:, 1:, :]` is never negative
when every value is non-negative, independent of any threshold applied
afterward).

**Decision: baselines train on the 40 non-ZC columns (`mav_*`,
`rms_*`, `wl_*`, `ssc_*`), not all 50.** This is a modeling time slice,
not a preprocessing change. The full 50-column matrix stays exactly as
saved in the `.npz` files, so the excluded columns remain there,
auditable, if anyone later wants to confirm they really are dead weight
rather than trust this document.

The reasoning is stronger than "uninformative, so drop it." A
zero-variance feature is a genuine numerical-stability risk for LDA
specifically. Its fit computes a (pseudo-)inverse of a covariance
matrix, and a column with zero variance across every single sample
contributes a literal zero row/column to that matrix, pushing it toward
singular. scikit-learn's implementation will not necessarily crash, but it
can silently degrade the fit in a way that is hard to trace back to "it is
the 10 dead columns" without already knowing they are there.

SSC is not affected by any of this. It measures direction reversals in
the signal's slope, which stays meaningful whether or not the signal is
rectified, confirmed directly in `tests/test_features.py`, where a
rectified synthetic signal correctly produces ZC=0 but SSC>0.

## Features computed

Per channel, per window: MAV (mean absolute value), RMS (root mean
square), WL (waveform length), ZC (zero crossings, see above, kept in
the saved data but excluded at modeling time), SSC (slope sign changes).
10 channels x 5 features = 50 columns, named `{feature}_{channel}` (e.g.
`mav_emg_3`), in a fixed order recorded alongside every saved matrix so
the columns can never silently drift from their names.

All formulas verified against hand-computed values in
`tests/test_features.py` before being trusted on real data, not just
run and eyeballed.

## ZC / SSC threshold

`zc_threshold` (`config.yaml`, `features:` section) is now moot given
confirmed rectification. No threshold value changes ZC's outcome, so
it is dead configuration, not a tunable parameter, for this dataset.

`ssc_threshold` remains a live, provisional default (0.01), reasoned
from the ~0.00244 quantization step visible in the very first raw sample
values seen in this project (roughly 4x that step, enough to filter
pure quantization noise while staying sensitive to genuine small slope
changes). Not yet validated against the real per-subject noise floor.
Revisit if per-class SSC distributions look off in a way that traces
back to threshold level noise.

## Label encoding

`rest` always maps to 0. Active labels sort numerically by
`(exercise, restimulus)`, not lexicographically as strings, which would
put `"2_10"` between `"2_1"` and `"2_2"`. Same class of bug already
caught and fixed in `src/data.py`'s `_sort_key`; applied proactively here
rather than needing to be rediscovered. Saved to
`data/processed/features/label_encoding.json`, sorted by index, so it is
human-readable and the exact mapping used is reproducible later (We
shall need this to label a confusion matrix with real gesture names
rather than bare integers).

Confirmed on the real data: 41 entries exactly, `rest`: 0, `2_1`
through `2_17`: 1-17, `3_1` through `3_23`: 18-40, in correct numeric
order throughout (`2_10` correctly sorts after `2_9`, not between `2_1`
and `2_2`).

## Split joining

Reuses `src.split.validate_subject_split` rather than
re-deriving overlap/exhaustiveness logic. Confirms the four subject
groups from `config.yaml` still form a valid partition of all 27
subjects before anything gets written to disk based on them.

## Output format

`data/processed/features/{train,validation,test,holdout}.npz`, each
containing `X` (n_windows, 50 - all columns, including the excluded ZC
ones, kept for auditability), `y` (n_windows,, integer class index),
`subject` (n_windows,), and `window_index` (n_windows,, traces back to
the row in `windows_metadata.parquet` this window came from, e.g. for
error analysis later, tracing a specific misclassified window back to
its original subject/gesture/repetition).

Real split sizes from the full 27-subject run: train 1,262,855 /
validation 209,740 / test 211,382 / holdout 206,796 (sums to
1,890,773, matching the previous total exactly).

## Validation: per-class feature distributions (real result, full 27-subject train split)

Run against `data/processed/features/train.npz` (1,262,855 windows, 41
classes). Full numbers, not eyeballed from the plot:

| | rest median | active median range | active median mean |
|---|---|---|---|
| MAV | 0.0322 | 0.1200 - 0.5194 | 0.2893 |
| RMS | 0.0335 | 0.1243 - 0.5386 | 0.2965 |

**Rest vs. active: cleanly separable.** Rest sits roughly an order of
magnitude below the floor of the active range on both features. The
~0.95% of rest windows that individually exceed the amplitude
threshold do not threaten this at all, since a brief spike barely moves a
20-sample window's *average*, even though it clearly moves its *peak*
(the same distinction already established when reconciling
run-level and window-level anomaly rates).

**Active-to-active: genuinely mixed, not uniformly separable.** Visual
inspection of `mav_by_class_train.png` / `rms_by_class_train.png` shows
real spread in central tendency (`2_6`, `2_14`, `3_20`, `3_22` sit near
the top of the range, roughly 3-4x the medians of `2_7`, `3_7`, `3_9`,
`3_14`, `3_15`, `3_17`), but a cluster of classes, `2_1`, `2_8`, `2_9`,
`2_11`, `2_12`, `3_1`, `3_5`, `3_6`, have near-identical medians with
heavily overlapping boxes. A classifier relying on amplitude alone
(MAV/RMS) should be expected to confuse pairs within that cluster
specifically, not perform uniformly across all 40 active classes. This
is realistic problem difficulty to plan around, not a pipeline
problem. If accuracy is noticeably worse on this specific cluster
of classes, that is confirmation of this finding, not a surprise to
debug from scratch.

**MAV and RMS are near-redundant on this data.** The two plots are
almost identical class-by-class, which is expected for a rectified,
consistently-shaped signal (RMS tracks MAV closely in that case) but
means these two features are carrying largely overlapping information
rather than two independent signals. WL and SSC were not part of this
required check but are the more likely candidates to do real
discriminative work between similar-amplitude active gestures, since
they capture waveform complexity and slope-reversal rate rather than
amplitude, worth the same per-class check as a natural follow-up,
though not a blocker for modeling.