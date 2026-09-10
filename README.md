# Subject-Independent Myoelectric Gesture Classification

## Overview

This research project investigates **subject-independent gesture classification for a myoelectric transradial prosthetic hand**. The aim is to classify functional hand and wrist gestures from multi-channel forearm surface electromyography (sEMG) for a person whose signals were not used to train the model.

The deployment scenario is intentionally demanding: a gesture classifier should generalize to a new prospective prosthesis user without relying on that person's repetitions appearing in both training and evaluation data. This avoids the overly optimistic results that can arise when a participant's individual EMG characteristics leak across dataset splits.

> This is a research and portfolio project. It is **not** a clinical device, a validated prosthetic control system, or evidence of safe real world deployment.

## Dataset

The project uses **Ninapro Database 1 (DB1)**, a public benchmark of forearm sEMG and hand-kinematic recordings from 27 participants with intact limbs performing 52 hand movements plus rest. Signals were recorded using ten Otto Bock MyoBock 13E200 electrodes at 100 Hz. The original database and acquisition protocol are described by Atzori et al. (2014).

The working dataset is accessed through a CSV mirror for practical processing, while the official Ninapro record remains the source of provenance and citation. The raw data are excluded from version control.

## Scope decision

This project uses all **27 participants** and focuses on movements that better reflect the control granularity of a practical myoelectric hand:

- **Exercise B:** 17 hand configuration and basic wrist movement classes
- **Exercise C:** 23 grasping and functional movement classes
- **Rest:** 1 class

This produces **41 classes in total**.

Exercise A, which contains individual finger movements, is deliberately excluded. Independent finger control is not the standard output of unmodified surface-EMG control in commercial myoelectric hands; whole hand grasp patterns and wrist relevant functions offer a more defensible functional target for this study. The exclusion is therefore a use case decision, rather than an attempt to simplify the benchmark arbitrarily.

## Evaluation protocol

Participants, not repetitions or individual windows, are separated across splits:

| Split | Participants | Purpose |
| --- | ---: | --- |
| Training | 18 | Fit model parameters |
| Validation | 3 | Select models and tune hyperparameters |
| Test | 3 | Single final held-out evaluation |
| Fully reserved | 3 | Optional, post development zero-shot generalization check |

No participant may occur in more than one split. The test set will not be used while selecting features, architectures, hyperparameters, or decision rules. The fully reserved participants will remain untouched until the optional final generalization analysis.

## Models

The study compares classical and learned representations of the same sEMG problem:

1. **Linear Discriminant Analysis (LDA):** a fast, domain-relevant baseline using time-domain features such as mean absolute value (MAV), root mean square (RMS), waveform length, zero crossings, and slope sign changes.
2. **Scalable classical baseline:** a linear classifier suitable for a large window level dataset, or a Random Forest comparison where computationally appropriate.
3. **Small 1D convolutional neural network (CNN):** a compact network trained on raw, windowed multi-channel sEMG to learn temporal features directly.

The planned preprocessing uses 200 ms windows with a 50 ms step. Any filtering, transition trimming, normalization, augmentation, and class-weighting choices will be fitted using training data only and documented before evaluation.

## Limitations and responsible interpretation

- **Participant population:** DB1 contains participants with intact limbs, not people with amputations. Residual limb EMG patterns, electrode placement, skin properties, fatigue, and control strategies may differ substantially for prosthesis users.
- **Sampling rate:** DB1 is sampled at **100 Hz**. This constrains the frequency information available compared with higher rate EMG acquisition systems.
- **Dataset shift:** performance on a benchmark dataset does not establish performance across new sensors, electrode placements, limb conditions, or daily life movement contexts.
- **Safety:** a misclassified command in a physical prosthesis could lead to an unintended grasp or release. Classification accuracy alone is not a safety validation.

Results will be reported as benchmark findings under this defined protocol, with uncertainty and per-class/per-participant errors considered alongside aggregate metrics.

## Reproducibility

- Raw and processed datasets, trained weights, and generated artifacts are not committed to the repository.
- Dataset source, file names, checksums, and access terms are recorded in a dataset manifest.
- Random seeds, split assignments, preprocessing parameters, and model settings are versioned in configuration files.
- Assertions will verify that there is zero participant overlap between splits.

## Reference

Atzori, M., Gijsberts, A., Heynen, S., et al. (2014). *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. **Scientific Data, 1**, 140053. https://doi.org/10.1038/sdata.2014.53

## Data provenance

- [Official Ninapro DB1 record](https://ninapro.hevs.ch/instructions/DB1.html)
- [Working CSV mirror on Kaggle](https://www.kaggle.com/datasets/mansibmursalin/ninapro-db1-full-dataset)

