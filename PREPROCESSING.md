# Preprocessing decisions

Recorded here because these are exactly the kind of choices that silently
determine downstream results if they are left undocumented.

## Window size and step

20 samples (200 ms) window, 5 samples (50 ms) step, at the dataset's
native 100 Hz sampling rate. Recorded in `config.yaml` under `windowing:`.

Both segment types in this dataset are far longer than one window (active
runs: 184 - 661 samples; rest runs: 227 - 10,937 samples. See
DATASET_MANIFEST.md), so this window size produces many windows per run
rather than being a binding constraint on segment length.

## Windows never cross a run boundary

Implemented via `src/segments.add_run_id`, extracted from the EDA
notebook's run identification logic rather than reimplemented from
scratch. A "run" is a maximal contiguous block sharing
`(subject, exercise, restimulus, rerepetition)`. `src/windows.make_windows`
groups by `run_id` and only ever slides a window within one group, so a
window can never mix samples from two different subjects, gestures, or
repetitions.

## Label resolution: not actually majority vote, a correction from the original plan

The original plan was to assign each window's label by majority vote,
since windows are confined to one run this should be unanimous almost always.
Writing the code made that framing wrong, not just imprecise:
because a run is *defined*, in part, by a single constant
`restimulus` value, every row inside a correctly constructed run confined
window is *always* unanimous. There is no real ambiguity left to vote on
by the time windowing runs. "almost always" should have been "always, by
construction."

So `_resolve_window_label` does not vote. It asserts unanimity and raises
`ValueError` if that is ever violated. A non-unanimous window is not a
legitimate edge case to resolve gracefully. It would mean run boundary
computation has a bug, and the correct response is to fail loudly and stop
trusting the output, not silently pick a label and continue. Tested
directly against both a unanimous and a deliberately non-unanimous
synthetic input in `tests/test_windows.py`.

## Rest-transient handling: flag, do not drop

Every window, active or rest, gets `peak_abs_amplitude` and
`peak_abs_amplitude_above_threshold` (default threshold 1.4, matching the
DATASET_MANIFEST.md rest-peak screening). No window is dropped during
preprocessing based on this value. This keeps preprocessing non-
destructive and auditable: whether/how to use the flag (exclude at
training time, weight differently, or ignore it) is left to modeling code
in a later day, not decided silently here.

Run level screening found 7.40% of rest RUNS have peak amplitude >=1.4
somewhere in their duration. Window level screening found only 0.95%
of rest WINDOWS individually exceed that threshold. These are
not inconsistent, they measure different units. A run can be flagged by
one brief spike while the large majority of its own windows remain
ordinary. Arithmetic on the full run confirms flagged windows are
concentrated in roughly 12-13% of each flagged run's own windows, not
spread throughout it. This matches the two brief, localized spikes
visible in the original raw-trace figure against an otherwise flat
109-second rest recording. The window-level figure (0.95%) is the more
directly relevant one for training decisions, since training operates on
windows, not runs.

## Trailing partial windows are dropped

If a run's remaining samples after the last full window are fewer than
`window_samples`, they are dropped, not padded. Simpler and more defensible
than padding with values that were never actually recorded.

## Row provenance

Output windows record `frame_row_start` / `frame_row_end`: positions
within the *filtered* dataframe passed into `make_windows` (i.e. after
`load_ninapro_csv`'s exercise filtering), not positions in the original
unfiltered csv. Anyone tracing further back needs their own record of how
that frame was produced.

## Known scale consideration before running the full 27-subject job

Rough arithmetic at window = 20/step = 5: on the order of 70-100+ windows per
run, across ~21,650 runs total (10,800 active + 10,854 rest), roughly
1.5-2M windows for the full dataset. Run against 2-3 subjects first to
confirm runtime and output size before committing to the full run.
