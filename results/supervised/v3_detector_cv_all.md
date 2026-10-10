# v3 detector: peer baselines, ensemble and persistent alerts, ALL users, user-level cross-validation (`all`)

99 users (69 insiders, 30 benign), 25495 user-days. 5 folds x 3 repeats, date-matched training, 1000 bootstrap resamples. Columns: {'v1': 40, 'v1_peer': 56, 'v2': 57, 'v3': 73}. Commit `2a752bd11dcb8216cf562a377ed0bdda41b1a5de`.

**This is NOT a clean unseen-user test.** The locked users were scored once by the frozen v1 detector, so every insider has been seen; this is development-style and optimistic. Read the PAIRED differences against v1; the frozen v1 locked result remains the clean evidence. Features, rules and quantiles were fixed before any result and ALL variants are shown. With only 30 benign users the false-alarm figures are coarse.

## 1. Day level: insiders only, date-stratified

#### all insiders: Gradient boosting (supervised)

Window labelling.

Computed on 46 of 69 calendar strata that contain both classes, covering 1357 of 1357 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 1 | 0.852 [0.788, 0.909] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 1 | 0.913 [0.865, 0.953] | +0.060 [+0.022, +0.100] * |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.931 [0.897, 0.959] | +0.077 [+0.043, +0.119] * |
| Gradient boosting (supervised), v3 (+ email + peer) | 1 | 0.948 [0.923, 0.969] | +0.093 [+0.053, +0.142] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### all insiders: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 46 of 69 calendar strata that contain both classes, covering 1357 of 1357 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (frozen features) | 1 | 0.909 [0.859, 0.948] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 1 | 0.922 [0.874, 0.959] | +0.014 [-0.021, +0.046] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.914 [0.865, 0.951] | +0.006 [-0.015, +0.029] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 1 | 0.937 [0.900, 0.965] | +0.027 [-0.007, +0.066] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### all insiders: Ensemble: mean rank of gradient boosting and GRU

Window labelling.

Computed on 46 of 69 calendar strata that contain both classes, covering 1357 of 1357 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) [95% CI] |
|---|---|---|---|
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 1 | 0.901 [0.848, 0.945] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 1 | 0.928 [0.885, 0.962] | +0.026 [-0.008, +0.056] |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 1 | 0.940 [0.904, 0.966] | +0.038 [+0.014, +0.068] * |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 1 | 0.955 [0.927, 0.975] | +0.052 [+0.018, +0.091] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 1 insiders only: Gradient boosting (supervised)

Window labelling.

Computed on 28 of 64 calendar strata that contain both classes, covering 149 of 149 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 1 | 0.989 [0.977, 0.996] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 1 | 0.989 [0.976, 0.996] | +0.001 [-0.008, +0.009] |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.985 [0.970, 0.995] | -0.003 [-0.010, +0.001] |
| Gradient boosting (supervised), v3 (+ email + peer) | 1 | 0.988 [0.974, 0.995] | -0.001 [-0.007, +0.004] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 1 insiders only: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 28 of 64 calendar strata that contain both classes, covering 149 of 149 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (frozen features) | 1 | 0.988 [0.973, 0.997] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 1 | 0.987 [0.968, 0.997] | -0.002 [-0.008, +0.003] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.986 [0.967, 0.996] | -0.003 [-0.008, +0.001] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 1 | 0.984 [0.958, 0.997] | -0.005 [-0.018, +0.002] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 1 insiders only: Ensemble: mean rank of gradient boosting and GRU

Window labelling.

Computed on 28 of 64 calendar strata that contain both classes, covering 149 of 149 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) [95% CI] |
|---|---|---|---|
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 1 | 0.991 [0.978, 0.998] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 1 | 0.992 [0.979, 0.998] | +0.001 [-0.002, +0.004] |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 1 | 0.990 [0.975, 0.997] | -0.001 [-0.005, +0.002] |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 1 | 0.989 [0.974, 0.997] | -0.002 [-0.007, +0.001] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 2 insiders only: Gradient boosting (supervised)

Window labelling.

Computed on 41 of 68 calendar strata that contain both classes, covering 1123 of 1188 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 1 | 0.634 [0.534, 0.731] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 1 | 0.804 [0.708, 0.885] | +0.169 [+0.083, +0.253] * |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.839 [0.777, 0.889] | +0.206 [+0.133, +0.284] * |
| Gradient boosting (supervised), v3 (+ email + peer) | 1 | 0.886 [0.840, 0.924] | +0.252 [+0.170, +0.334] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 2 insiders only: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 41 of 68 calendar strata that contain both classes, covering 1123 of 1188 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (frozen features) | 1 | 0.805 [0.718, 0.882] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 1 | 0.844 [0.758, 0.910] | +0.041 [-0.038, +0.112] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.824 [0.731, 0.898] | +0.021 [-0.032, +0.070] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 1 | 0.891 [0.834, 0.935] | +0.087 [+0.015, +0.158] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 2 insiders only: Ensemble: mean rank of gradient boosting and GRU

Window labelling.

Computed on 41 of 68 calendar strata that contain both classes, covering 1123 of 1188 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) [95% CI] |
|---|---|---|---|
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 1 | 0.761 [0.660, 0.845] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 1 | 0.840 [0.752, 0.909] | +0.081 [+0.003, +0.144] * |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 1 | 0.862 [0.794, 0.916] | +0.104 [+0.050, +0.161] * |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 1 | 0.909 [0.857, 0.948] | +0.151 [+0.075, +0.224] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 3 insiders only: Gradient boosting (supervised)

Window labelling.

Computed on 10 of 69 calendar strata that contain both classes, covering 20 of 20 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 1 | 0.979 [0.922, 1.000] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 1 | 0.924 [0.819, 1.000] | -0.051 [-0.151, +0.004] |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.975 [0.916, 1.000] | -0.008 [-0.056, +0.026] |
| Gradient boosting (supervised), v3 (+ email + peer) | 1 | 0.962 [0.897, 1.000] | -0.016 [-0.076, +0.022] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 3 insiders only: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 10 of 69 calendar strata that contain both classes, covering 20 of 20 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (frozen features) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (frozen features) | 1 | 0.929 [0.791, 1.000] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 1 | 0.878 [0.716, 1.000] | -0.047 [-0.148, +0.000] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.899 [0.733, 1.000] | -0.029 [-0.106, +0.011] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 1 | 0.855 [0.672, 1.000] | -0.070 [-0.193, +0.000] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 3 insiders only: Ensemble: mean rank of gradient boosting and GRU

Window labelling.

Computed on 10 of 69 calendar strata that contain both classes, covering 20 of 20 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) [95% CI] |
|---|---|---|---|
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 1 | 0.975 [0.923, 1.000] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 1 | 0.916 [0.815, 1.000] | -0.054 [-0.145, +0.000] |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 1 | 0.977 [0.919, 1.000] | +0.001 [-0.049, +0.062] |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 1 | 0.941 [0.852, 1.000] | -0.032 [-0.119, +0.018] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

## 2. Whole-timeline person ranking (user-level AUROC, insiders vs benign users)

#### rule: top_k_mean

| Model | user-level AUROC [95% CI] | minus v1 [95% CI] |
|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 0.893 [0.823, 0.947] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 0.922 [0.856, 0.971] | +0.030 [-0.024, +0.092] |
| Gradient boosting (supervised), v2 (+ email) | 0.887 [0.808, 0.948] | -0.006 [-0.038, +0.023] |
| Gradient boosting (supervised), v3 (+ email + peer) | 0.922 [0.852, 0.971] | +0.030 [-0.031, +0.097] |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 0.955 [0.901, 0.991] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 0.932 [0.866, 0.978] | -0.024 [-0.075, +0.018] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 0.952 [0.889, 0.993] | -0.003 [-0.032, +0.030] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 0.889 [0.804, 0.953] | -0.066 [-0.143, +0.000] |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 0.957 [0.910, 0.989] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 0.963 [0.914, 0.994] | +0.006 [-0.037, +0.050] |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 0.953 [0.902, 0.989] | -0.004 [-0.021, +0.014] |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 0.941 [0.884, 0.982] | -0.016 [-0.064, +0.030] |

#### rule: max

| Model | user-level AUROC [95% CI] | minus v1 [95% CI] |
|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 0.957 [0.919, 0.986] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 0.971 [0.930, 0.997] | +0.014 [-0.024, +0.052] |
| Gradient boosting (supervised), v2 (+ email) | 0.959 [0.913, 0.993] | +0.001 [-0.038, +0.036] |
| Gradient boosting (supervised), v3 (+ email + peer) | 0.954 [0.904, 0.990] | -0.004 [-0.047, +0.040] |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 0.974 [0.928, 1.000] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 0.969 [0.927, 0.995] | -0.006 [-0.042, +0.030] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 0.972 [0.917, 1.000] | -0.002 [-0.023, +0.020] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 0.934 [0.864, 0.984] | -0.042 [-0.114, +0.020] |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 0.989 [0.973, 1.000] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 0.989 [0.967, 1.000] | -0.000 [-0.021, +0.018] |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 0.986 [0.955, 1.000] | -0.004 [-0.027, +0.013] |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 0.986 [0.952, 1.000] | -0.004 [-0.037, +0.020] |

## 3. Matched-period check (insider's window vs benign users on the same dates)

| Model | matched-period AUROC [95% CI] | minus v1 [95% CI] |
|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 0.995 [0.984, 1.000] | (reference) |
| Gradient boosting (supervised), v1 + peer baselines | 0.989 [0.973, 0.999] | -0.006 [-0.019, +0.004] |
| Gradient boosting (supervised), v2 (+ email) | 0.998 [0.991, 1.000] | +0.003 [-0.002, +0.012] |
| Gradient boosting (supervised), v3 (+ email + peer) | 0.997 [0.990, 1.000] | +0.002 [-0.004, +0.011] |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 0.997 [0.990, 1.000] | (reference) |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 0.994 [0.980, 1.000] | -0.004 [-0.015, +0.003] |
| GRU over the previous 14 days (supervised), v2 (+ email) | 0.993 [0.976, 1.000] | -0.004 [-0.018, +0.002] |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 0.985 [0.957, 1.000] | -0.013 [-0.040, +0.002] |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 0.998 [0.993, 1.000] | (reference) |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 0.995 [0.983, 1.000] | -0.003 [-0.015, +0.006] |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 0.998 [0.992, 1.000] | -0.000 [-0.003, +0.002] |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 0.996 [0.985, 1.000] | -0.002 [-0.014, +0.006] |

## 4. Alert simulation with CROSS-FITTED cutoffs (a fold of users is judged with a cutoff set on benign users of the other folds)

#### rule `single`, cutoff = 99th percentile of benign day scores

| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | insiders alerted BEFORE window | benign users alerted | alert days per benign user |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 67 of 69 | 29 of 29 | 30 of 30 | 8 of 10 | 0 | 43 of 69 | 17 of 30 | 3.9 |
| Gradient boosting (supervised), v1 + peer baselines | 65 of 69 | 29 of 29 | 29 of 30 | 7 of 10 | 0 | 27 of 69 | 8 of 30 | 6.0 |
| Gradient boosting (supervised), v2 (+ email) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 37 of 69 | 19 of 30 | 3.7 |
| Gradient boosting (supervised), v3 (+ email + peer) | 67 of 69 | 29 of 29 | 30 of 30 | 8 of 10 | 0 | 35 of 69 | 11 of 30 | 6.6 |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 67 of 69 | 28 of 29 | 30 of 30 | 9 of 10 | 0 | 42 of 69 | 8 of 30 | 4.3 |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 64 of 69 | 27 of 29 | 30 of 30 | 7 of 10 | 0 | 34 of 69 | 4 of 30 | 8.0 |
| GRU over the previous 14 days (supervised), v2 (+ email) | 65 of 69 | 28 of 29 | 29 of 30 | 8 of 10 | 0 | 44 of 69 | 5 of 30 | 6.5 |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 56 of 69 | 23 of 29 | 26 of 30 | 7 of 10 | 0 | 20 of 69 | 2 of 30 | 12.1 |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 51 of 69 | 11 of 30 | 3.6 |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 66 of 69 | 29 of 29 | 30 of 30 | 7 of 10 | 0 | 34 of 69 | 6 of 30 | 11.0 |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 49 of 69 | 12 of 30 | 4.7 |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 65 of 69 | 28 of 29 | 28 of 30 | 9 of 10 | 0 | 32 of 69 | 6 of 30 | 9.3 |

#### rule `single`, cutoff = 95th percentile of benign day scores

| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | insiders alerted BEFORE window | benign users alerted | alert days per benign user |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 57 of 69 | 24 of 30 | 18.4 |
| Gradient boosting (supervised), v1 + peer baselines | 67 of 69 | 29 of 29 | 30 of 30 | 8 of 10 | 0 | 51 of 69 | 23 of 30 | 21.1 |
| Gradient boosting (supervised), v2 (+ email) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 57 of 69 | 28 of 30 | 17.9 |
| Gradient boosting (supervised), v3 (+ email + peer) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 56 of 69 | 26 of 30 | 18.0 |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 63 of 69 | 21 of 30 | 18.2 |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 54 of 69 | 14 of 30 | 19.3 |
| GRU over the previous 14 days (supervised), v2 (+ email) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 64 of 69 | 24 of 30 | 20.2 |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 67 of 69 | 28 of 29 | 30 of 30 | 9 of 10 | 0 | 45 of 69 | 9 of 30 | 17.5 |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 63 of 69 | 21 of 30 | 18.0 |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 62 of 69 | 19 of 30 | 18.7 |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 63 of 69 | 25 of 30 | 18.4 |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 59 of 69 | 22 of 30 | 17.5 |

#### rule `persist_2of3`, cutoff = 99th percentile of benign day scores

| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | insiders alerted BEFORE window | benign users alerted | alert days per benign user |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 64 of 69 | 28 of 29 | 28 of 30 | 8 of 10 | 1 | 32 of 69 | 8 of 30 | 1.4 |
| Gradient boosting (supervised), v1 + peer baselines | 61 of 69 | 26 of 29 | 29 of 30 | 6 of 10 | 1 | 19 of 69 | 5 of 30 | 4.6 |
| Gradient boosting (supervised), v2 (+ email) | 65 of 69 | 27 of 29 | 30 of 30 | 8 of 10 | 1 | 27 of 69 | 7 of 30 | 1.0 |
| Gradient boosting (supervised), v3 (+ email + peer) | 63 of 69 | 27 of 29 | 30 of 30 | 6 of 10 | 1 | 24 of 69 | 8 of 30 | 4.9 |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 65 of 69 | 27 of 29 | 30 of 30 | 8 of 10 | 1 | 37 of 69 | 4 of 30 | 3.4 |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 59 of 69 | 25 of 29 | 27 of 30 | 7 of 10 | 1 | 29 of 69 | 3 of 30 | 6.7 |
| GRU over the previous 14 days (supervised), v2 (+ email) | 63 of 69 | 26 of 29 | 29 of 30 | 8 of 10 | 1 | 35 of 69 | 3 of 30 | 5.3 |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 50 of 69 | 19 of 29 | 26 of 30 | 5 of 10 | 2 | 14 of 69 | 2 of 30 | 11.3 |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 40 of 69 | 6 of 30 | 2.3 |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 61 of 69 | 26 of 29 | 29 of 30 | 6 of 10 | 1 | 29 of 69 | 2 of 30 | 9.9 |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 67 of 69 | 29 of 29 | 30 of 30 | 8 of 10 | 0 | 41 of 69 | 2 of 30 | 3.4 |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 60 of 69 | 26 of 29 | 28 of 30 | 6 of 10 | 1 | 24 of 69 | 2 of 30 | 8.5 |

#### rule `persist_2of3`, cutoff = 95th percentile of benign day scores

| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | insiders alerted BEFORE window | benign users alerted | alert days per benign user |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting (supervised), v1 (frozen features) | 67 of 69 | 29 of 29 | 30 of 30 | 8 of 10 | 0 | 47 of 69 | 16 of 30 | 11.4 |
| Gradient boosting (supervised), v1 + peer baselines | 65 of 69 | 28 of 29 | 30 of 30 | 7 of 10 | 1 | 41 of 69 | 19 of 30 | 17.3 |
| Gradient boosting (supervised), v2 (+ email) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 48 of 69 | 19 of 30 | 10.1 |
| Gradient boosting (supervised), v3 (+ email + peer) | 66 of 69 | 28 of 29 | 30 of 30 | 8 of 10 | 0 | 49 of 69 | 22 of 30 | 13.0 |
| GRU over the previous 14 days (supervised), v1 (frozen features) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 56 of 69 | 19 of 30 | 13.9 |
| GRU over the previous 14 days (supervised), v1 + peer baselines | 65 of 69 | 27 of 29 | 29 of 30 | 9 of 10 | 1 | 47 of 69 | 8 of 30 | 17.7 |
| GRU over the previous 14 days (supervised), v2 (+ email) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 54 of 69 | 19 of 30 | 15.3 |
| GRU over the previous 14 days (supervised), v3 (+ email + peer) | 61 of 69 | 26 of 29 | 27 of 30 | 8 of 10 | 1 | 35 of 69 | 7 of 30 | 16.2 |
| Ensemble: mean rank of gradient boosting and GRU, v1 (frozen features) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 56 of 69 | 15 of 30 | 13.0 |
| Ensemble: mean rank of gradient boosting and GRU, v1 + peer baselines | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 54 of 69 | 14 of 30 | 15.9 |
| Ensemble: mean rank of gradient boosting and GRU, v2 (+ email) | 69 of 69 | 29 of 29 | 30 of 30 | 10 of 10 | 0 | 56 of 69 | 21 of 30 | 12.3 |
| Ensemble: mean rank of gradient boosting and GRU, v3 (+ email + peer) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 51 of 69 | 15 of 30 | 13.9 |

_All users, cross-validated: development-style numbers, optimistic. Peer baselines use staff role/team from LDAP and the activity of all employees; email features are envelope data only._
