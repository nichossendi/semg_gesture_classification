# Dataset Manifest - Ninapro DB1

This manifest records the inspected Kaggle csv working copy used for sEMG gesture classification. Raw participant data, processed windows, and model artefacts are excluded from version control.

## Dataset identity

| Field | Value |
| --- | --- |
| Dataset name | Ninapro Database 1 (DB1) |
| Dataset version / release date | Kaggle mirror metadata not independently versioned; official DB1 record cited below |
| Official source | https://ninapro.hevs.ch/instructions/DB1.html |
| Working CSV mirror | https://www.kaggle.com/datasets/mansibmursalin/ninapro-db1-full-dataset |
| Original dataset citation | Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Mittaz Hager, A., Elsig, S., Giatsidis, G., Bassetto, F., Müller, H., et al. (2014). *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. *Scientific Data, 1*, 140053. https://doi.org/10.1038/sdata.2014.53 |
| Intended project use | Subject-independent research evaluation of myoelectric gesture classification, not clinical validation |

## Download record

| Field | Value |
| --- | --- |
| Downloaded by | Local project user |
| Download date (UTC) | 2026-09-09 |
| Source used for working copy | Kaggle CSV mirror listed above |
| Source account / access method | Kaggle download, account details not recorded |
| License / access terms reviewed | Repo is currently private. Terms not yet confirmed against Kaggle's stated license — must be verified before any public release of this repository. |
| Local raw file path | `data/raw/Ninapro_DB1.csv`, not committed |
| File name | `Ninapro_DB1.csv` |
| File size (bytes) | 3,371,688,405 |
| SHA-256 checksum | 3a0d21a7fdb1520dfaf29b281bd4a8bd03782062a1d6fd574c7e194cbde79ed4 |

## Expected DB1 data characteristics

| Characteristic | Expected value | Verified value |
| --- | ---: | --- |
| Participants | 27 intact-limbed participants | 27 unique IDs (`1`-`27`) |
| Acquisition rate | 100 Hz | 100 Hz, used to convert sample counts to durations |
| sEMG channels | 10 Otto Bock MyoBock 13E200 channels | 10 columns: `emg_0`-`emg_9` |
| Exercises | A, B, C | IDs `1`, `2`, `3` present |
| Movements | 52 movements plus rest | Consistent with DB1 documentation, csv labels reset by exercise, so 52 cannot be inferred from globally unique label values |
| Exercise B classes | 17 | 17 non-rest `restimulus` labels in Exercise 2 |
| Exercise C classes | 23 | 23 non-rest `restimulus` labels in Exercise 3 |
| Project target classes | Exercise B + Exercise C + rest = 41 | 41 EDA plotting labels after unifying rest |

## Schema and quality checks

| Check | Result |
| --- | --- |
| Column names and data types recorded | 38 columns: 10 `float64` EMG, 22 `float64` glove, and 6 `int64` metadata columns; full schema saved in `reports/eda_loader_report.txt` |
| EMG columns identified | `emg_0`-`emg_9` (10 columns) |
| Participant identifier column confirmed | `subject` |
| Exercise identifier column confirmed | `exercise` |
| `stimulus` and `restimulus` label columns confirmed | Both present; `restimulus` used for EDA because it is the refined label |
| Repetition identifier columns confirmed | `repetition` and `rerepetition` present |
| Row count | 12,553,611 full csv rows, 9,822,218 rows after filtering Exercises B and C |
| Missing-value summary | No missing values in any of the 38 columns |
| Signal/dropout anomalies | No timestamp column, so dropped/duplicated samples cannot be proven directly. Active refined-label runs range from 184 to 661 samples (1.84 - 6.61s); 107/10,800 lie outside 2 - 6 s. Adjacent-row check: 1.34% of rows have all 10 EMG channels identical to the immediately preceding row. This is plausibly concentrated in near zero amplitude rest periods (consistent with quantization at low amplitude, as seen directly in the raw trace figures) rather than sensor dropout, but the rest vs active split of this rate has not yet been computed to confirm that. Treat the interpretation as open, not resolved. Recorded as timing/duplication variability, not as confirmed corruption. |
| Exercise B/C label convention verified | `restimulus` values 1 - 17 are reused in both exercises. Use `(exercise, restimulus)` for active labels |
| Label used for modeling (`restimulus` expected) | `restimulus` |

## EDA findings: repetitions, timing, and balance

### Repetition presence is complete, but it is not a segment quality guarantee

Exercise B (17) + Exercise C (23) gives 40 active movements. At 10 nominal repetitions per movement, the expected total is **400 active labeled repetitions per subject**. The contiguous run count is exactly 400 for every participant: **min = 400, max = 400, mean = 400.0**.

This validates label presence, not signal quality. A truncated or noisy segment still contributes one run if its label is present. Duration must therefore be evaluated separately before downstream windowing.

### Timing summary from contiguous `(subject, exercise, restimulus, rerepetition)` runs

| Segment type | Count | Samples: min / median / mean / max | Duration at 100 Hz: min / median / mean / max |
| --- | ---: | --- | --- |
| Active | 10,800 | 184 / 378 / 377.18 / 661 | 1.84 / 3.78 / 3.772 / 6.61 s |
| Rest | 10,854 | 227 / 465 / 529.64 / 10,937 | 2.27 / 4.65 / 5.296 / 109.37 s |

The refined label durations are not fixed 5 second movement intervals. The observed range should not be treated as proof of dropped samples without timestamps. Long rest runs, including the 109.37 second maximum, are plausible because `rerepetition == 0` collects unlabeled/baseline periods rather than only inter movement gaps.

### Class balance

The unified rest EDA table contains 9,822,218 samples:

- Rest: 5,748,692 samples (**58.53%**)
- All 40 active gesture labels combined: 4,073,526 samples (**41.47%**)
- Rest to active sample ratio: **1.41:1**
- Individual active label range: 85,442 - 124,115 samples, coefficient of variation: 7.93%

Rest dominates every individual gesture bar because it is one label compared against 40 separate active labels, but it does not dominate the recording time by a factor of 50. The active labels are comparatively uniform. Later modelling must nevertheless handle the rest class deliberately (for example, with class weighting or rest window subsampling).

### Rest-labeled transient activity

Rest labeled segments show systematic high amplitude events, not a single anomalous example. Across 10,854 rest runs, the maximum absolute amplitude per run has median 0.5688, 95th percentile 1.6146, and 99th percentile 2.7796. **1,939 runs (17.86%)** reach at least 1.0, **803 (7.40%)** reach at least 1.4, and each threshold occurs in all 27 subjects. This is a screening result, not a clinical interpretation of the events.

## Scope and split record

| Field | Value |
| --- | --- |
| Included exercises | B and C |
| Excluded exercise | A - individual finger movements are outside this project's whole hand and wrist relevant control scope |
| Rest handling | One rest class for EDA balance. Exercise specific combined labels retained only to diagnose active label collisions |
| Subject inclusion decision | All 27 retained for now. Segment count completeness alone is not used to claim signal quality. |
| Training participants | Pending: record IDs after leakage safe split assignment |
| Validation participants | Pending: record IDs after leakage safe split assignment |
| Test participants | Pending: record IDs after leakage safe split assignment |
| Fully reserved participants | Pending: record IDs after leakage safe split assignment |
| Random seed used to assign split | Pending |
| Participant-overlap assertion passed | Pending |

## Notes and deviations

- The csv mirror contains the expected DB1 fields but no explicit timestamp column.
- Raw `restimulus` values are not globally unique across Exercises B and C. EDA uses `(exercise, restimulus)` for active label inspection.
- No participant is excluded based on EDA. Timing variability and rest-labeled transient activity are retained as documented risks for preprocessing and modelling decisions.
