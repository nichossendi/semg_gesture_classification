## Scope and split record

| Field | Value |
| --- | --- |
| Included exercises | B and C |
| Excluded exercise | A - individual finger movements are outside this project's whole hand and wrist relevant control scope. |
| Rest handling | One rest class for EDA balance. Exercise specific combined labels are retained only to diagnose active label collisions. |
| Subject inclusion decision | All 27 participants retained. The label completeness check found 400 active labeled runs per participant. This does not itself establish segment quality. |
| Split assignment method | One seeded `numpy.random.default_rng(seed).permutation` over all 27 participant IDs, then sliced 18 / 3 / 3 / 3. Holdout is drawn in the same permutation, not selected from leftovers. |
| Random seed used to assign split | `20260907` |
| Training participants | 7, 24, 15, 27, 17, 25, 12, 26, 2, 21, 8, 3, 1, 22, 6, 9, 10, 5 |
| Validation participants | 20, 23, 16 |
| Test participants | 11, 14, 19 |
| Fully reserved participants | 4, 13, 18 |
| Participant-overlap assertion passed | Yes. The four groups are mutually exclusive and exhaustive for IDs 1-27. |
| Determinism test passed | Yes. Two calls with seed `20260907` produce the same assignment. |
| Rest-peak anomaly review | Screening threshold: rest-run peak absolute amplitude ≥1.4. Rates: train 6.40%, validation 4.06%, test 11.19%, holdout 12.94%, global 7.40%. Higher anomaly participants are distributed across train, test, and holdout. They are not concentrated solely in test. |
| Split decision | Keep seed `20260907`. The test and holdout rates are higher than train, which is documented as a limitation, but the split was fixed before this review and was not silently rerolled after inspecting EDA. If test-set accuracy comes out lower than validation accuracy later, part of that gap may be explained by test containing noisier subjects, not by worse generalization. |

