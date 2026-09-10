# Dataset Manifest - Ninapro DB1

Use this file to document exactly which dataset files were used for each experiment. Complete every field marked **Pending** after downloading and inspecting the data. Do not commit raw participant data, processed windows, trained model weights, or files containing personal identifiers.

## Dataset identity

| Field | Value |
| --- | --- |
| Dataset name | Ninapro Database 1 (DB1) |
| Dataset version / release date | **Pending** |
| Official source | https://ninapro.hevs.ch/instructions/DB1.html |
| Working CSV mirror | https://www.kaggle.com/datasets/mansibmursalin/ninapro-db1-full-dataset |
| Original dataset citation | Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Mittaz Hager, A., Elsig, S., Giatsidis, G., Bassetto, F., Müller, H., et al. (2014). *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. *Scientific Data, 1*, 140053. https://doi.org/10.1038/sdata.2014.53 |
| Intended project use | Subject-independent research evaluation of myoelectric gesture classification; not clinical validation |

## Download record

| Field | Value |
| --- | --- |
| Downloaded by | **Pending** |
| Download date (UTC) | **Pending** |
| Source used for working copy | **Pending** |
| Source account / access method | **Pending** |
| License / access terms reviewed | **Pending** |
| Local raw file path | `data/raw/` — do not commit the file |
| File name | **Pending** |
| File size (bytes) | **Pending** |
| SHA-256 checksum | **Pending** |

## Expected DB1 data characteristics

These values are taken from the official DB1 documentation and must be checked against the downloaded working copy before analysis.

| Characteristic | Expected value | Verified value |
| --- | ---: | --- |
| Participants | 27 intact-limbed participants | **Pending** |
| Acquisition rate | 100 Hz | **Pending** |
| sEMG channels | 10 Otto Bock MyoBock 13E200 channels | **Pending** |
| Exercises | A, B, C | **Pending** |
| Movements | 52 movements plus rest | **Pending** |
| Exercise B classes | 17 | **Pending** |
| Exercise C classes | 23 | **Pending** |
| Project target classes | Exercise B + Exercise C + rest = 41 | **Pending** |

## Schema and quality checks

Record the results from the initial inspection script or notebook.

| Check | Result |
| --- | --- |
| Column names and data types recorded | **Pending** |
| EMG columns identified | **Pending** |
| Participant identifier column confirmed | **Pending** |
| Exercise identifier column confirmed | **Pending** |
| `stimulus` and `restimulus` label columns confirmed | **Pending** |
| Repetition identifier columns confirmed | **Pending** |
| Row count | **Pending** |
| Missing-value summary | **Pending** |
| Signal/dropout anomalies | **Pending** |
| Exercise B/C label convention verified | **Pending** |
| Label used for modeling (`restimulus` expected) | **Pending** |

## Scope and split record

| Field | Value |
| --- | --- |
| Included exercises | B and C |
| Excluded exercise | A — individual-finger movements are outside this project's whole-hand and wrist-relevant control scope |
| Rest handling | One rest class; exact source and remapping rule to be documented after schema inspection |
| Training participants | **Pending: record participant IDs** |
| Validation participants | **Pending: record participant IDs** |
| Test participants | **Pending: record participant IDs** |
| Fully reserved participants | **Pending: record participant IDs** |
| Random seed used to assign split | **Pending** |
| Participant-overlap assertion passed | **Pending** |

## Notes and deviations

Add any differences between the CSV mirror and the official DB1 description here, including renamed columns, dropped variables, label remapping, or data-cleaning decisions. Each deviation must be explained and linked to the script or notebook that implements it.

**Pending**

