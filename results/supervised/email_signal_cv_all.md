# Email signal: v2 (+ email) against v1, ALL users, user-level cross-validation (`all`)

99 users (69 insiders, 30 benign), 25495 user-days. 5 folds x 3 repeats, date-matched training, 1000 bootstrap resamples. v1 = the frozen detector's 40 input columns; v2 = v1 + 17 email columns. Commit `ce8e57603e57c32588d84edab9acf3d6dd1edaeb`.

**This is NOT a clean unseen-user test.** The locked users were scored once by the frozen v1 detector, so every insider has been seen; this uses all users in user-level cross-validation (no user is scored by a model that trained on them) and is development-style and optimistic. Read the PAIRED differences v2 minus v1 (same folds, rows and resamples); the frozen v1 locked result remains the clean evidence. The email feature list was fixed in advance from the dataset readme and scenarios.txt.

## 1. Day level: insiders only, date-stratified (window days vs the same insiders' other days, same week)

#### all insiders: Gradient boosting (supervised)

Window labelling.

Computed on 46 of 69 calendar strata that contain both classes, covering 1357 of 1357 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 1 | 0.852 [0.788, 0.909] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.931 [0.897, 0.959] | +0.077 [+0.043, +0.119] * |
| Gradient boosting (supervised), email features only | 1 | 0.766 [0.702, 0.818] | -0.092 [-0.172, -0.006] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 1 | 0.522 [0.485, 0.558] | -0.334 [-0.403, -0.261] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 1 | 0.469 [0.387, 0.548] | -0.388 [-0.489, -0.281] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### all insiders: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 46 of 69 calendar strata that contain both classes, covering 1357 of 1357 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (no email) | 1 | 0.909 [0.859, 0.948] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.914 [0.865, 0.951] | +0.006 [-0.015, +0.029] |
| GRU over the previous 14 days (supervised), email features only | 1 | 0.811 [0.756, 0.853] | -0.101 [-0.172, -0.027] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 1 | 0.523 [0.460, 0.587] | -0.388 [-0.468, -0.305] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 1 | 0.555 [0.441, 0.662] | -0.356 [-0.469, -0.248] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 1 insiders only: Gradient boosting (supervised)

Window labelling.

Computed on 28 of 64 calendar strata that contain both classes, covering 149 of 149 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 1 | 0.989 [0.977, 0.996] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.985 [0.970, 0.995] | -0.003 [-0.010, +0.001] |
| Gradient boosting (supervised), email features only | 1 | 0.477 [0.389, 0.575] | -0.512 [-0.597, -0.416] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 1 | 0.452 [0.412, 0.502] | -0.535 [-0.580, -0.481] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 1 | 0.307 [0.245, 0.378] | -0.676 [-0.744, -0.606] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 1 insiders only: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 28 of 64 calendar strata that contain both classes, covering 149 of 149 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (no email) | 1 | 0.988 [0.973, 0.997] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.986 [0.967, 0.996] | -0.003 [-0.008, +0.001] |
| GRU over the previous 14 days (supervised), email features only | 1 | 0.563 [0.445, 0.671] | -0.428 [-0.544, -0.319] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 1 | 0.360 [0.245, 0.487] | -0.624 [-0.743, -0.504] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 1 | 0.514 [0.385, 0.614] | -0.477 [-0.601, -0.377] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 2 insiders only: Gradient boosting (supervised)

Window labelling.

Computed on 41 of 68 calendar strata that contain both classes, covering 1123 of 1188 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 1 | 0.634 [0.534, 0.731] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.839 [0.777, 0.889] | +0.206 [+0.133, +0.284] * |
| Gradient boosting (supervised), email features only | 1 | 0.799 [0.736, 0.856] | +0.165 [+0.066, +0.270] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 1 | 0.541 [0.495, 0.586] | -0.092 [-0.203, +0.029] |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 1 | 0.502 [0.411, 0.596] | -0.131 [-0.292, +0.031] |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 2 insiders only: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 41 of 68 calendar strata that contain both classes, covering 1123 of 1188 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (no email) | 1 | 0.805 [0.718, 0.882] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.824 [0.731, 0.898] | +0.021 [-0.032, +0.070] |
| GRU over the previous 14 days (supervised), email features only | 1 | 0.852 [0.793, 0.897] | +0.048 [-0.059, +0.161] |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 1 | 0.557 [0.484, 0.636] | -0.243 [-0.352, -0.138] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 1 | 0.581 [0.446, 0.714] | -0.222 [-0.356, -0.086] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 3 insiders only: Gradient boosting (supervised)

Window labelling.

Computed on 10 of 69 calendar strata that contain both classes, covering 20 of 20 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 1 | 0.979 [0.922, 1.000] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 1 | 0.975 [0.916, 1.000] | -0.008 [-0.056, +0.026] |
| Gradient boosting (supervised), email features only | 1 | 0.826 [0.700, 0.932] | -0.157 [-0.289, -0.038] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 1 | 0.532 [0.387, 0.684] | -0.442 [-0.595, -0.303] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 1 | 0.510 [0.256, 0.718] | -0.474 [-0.725, -0.268] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

#### scenario 3 insiders only: GRU over the previous 14 days (supervised)

Window labelling.

Computed on 10 of 69 calendar strata that contain both classes, covering 20 of 20 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus GRU over the previous 14 days (supervised), v1 (no email) [95% CI] |
|---|---|---|---|
| GRU over the previous 14 days (supervised), v1 (no email) | 1 | 0.929 [0.791, 1.000] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 1 | 0.899 [0.733, 1.000] | -0.029 [-0.106, +0.011] |
| GRU over the previous 14 days (supervised), email features only | 1 | 0.674 [0.494, 0.896] | -0.235 [-0.437, -0.060] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 1 | 0.546 [0.333, 0.709] | -0.397 [-0.625, -0.174] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 1 | 0.403 [0.184, 0.653] | -0.523 [-0.756, -0.301] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

## 2. Whole-timeline person ranking (user-level AUROC, insiders vs benign users)

#### rule: top_k_mean

| Model | user-level AUROC [95% CI] | minus v1 [95% CI] |
|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 0.893 [0.823, 0.947] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 0.887 [0.808, 0.948] | -0.006 [-0.038, +0.023] |
| Gradient boosting (supervised), email features only | 0.782 [0.684, 0.866] | -0.112 [-0.211, -0.001] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 0.115 [0.039, 0.204] | -0.777 [-0.864, -0.673] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 0.256 [0.128, 0.386] | -0.635 [-0.773, -0.492] * |
| GRU over the previous 14 days (supervised), v1 (no email) | 0.955 [0.901, 0.991] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 0.952 [0.889, 0.993] | -0.003 [-0.032, +0.030] |
| GRU over the previous 14 days (supervised), email features only | 0.758 [0.658, 0.846] | -0.198 [-0.301, -0.100] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 0.306 [0.187, 0.435] | -0.649 [-0.779, -0.512] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 0.535 [0.388, 0.678] | -0.420 [-0.576, -0.265] * |

#### rule: max

| Model | user-level AUROC [95% CI] | minus v1 [95% CI] |
|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 0.957 [0.919, 0.986] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 0.959 [0.913, 0.993] | +0.001 [-0.038, +0.036] |
| Gradient boosting (supervised), email features only | 0.815 [0.722, 0.888] | -0.147 [-0.242, -0.058] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 0.400 [0.291, 0.511] | -0.558 [-0.666, -0.438] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 0.324 [0.189, 0.469] | -0.632 [-0.772, -0.488] * |
| GRU over the previous 14 days (supervised), v1 (no email) | 0.974 [0.928, 1.000] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 0.972 [0.917, 1.000] | -0.002 [-0.023, +0.020] |
| GRU over the previous 14 days (supervised), email features only | 0.767 [0.663, 0.854] | -0.209 [-0.316, -0.118] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 0.312 [0.195, 0.440] | -0.662 [-0.791, -0.532] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 0.540 [0.390, 0.685] | -0.435 [-0.591, -0.280] * |

## 3. Matched-period check (insider's window vs benign users on the same dates)

| Model | matched-period AUROC [95% CI] | minus v1 [95% CI] |
|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 0.995 [0.984, 1.000] | (reference) |
| Gradient boosting (supervised), v2 (+ email) | 0.998 [0.991, 1.000] | +0.003 [-0.002, +0.012] |
| Gradient boosting (supervised), email features only | 0.897 [0.831, 0.952] | -0.099 [-0.163, -0.043] * |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 0.323 [0.236, 0.421] | -0.672 [-0.756, -0.577] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 0.463 [0.339, 0.591] | -0.530 [-0.658, -0.405] * |
| GRU over the previous 14 days (supervised), v1 (no email) | 0.997 [0.990, 1.000] | (reference) |
| GRU over the previous 14 days (supervised), v2 (+ email) | 0.993 [0.976, 1.000] | -0.004 [-0.018, +0.002] |
| GRU over the previous 14 days (supervised), email features only | 0.881 [0.817, 0.935] | -0.117 [-0.179, -0.062] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 0.527 [0.390, 0.659] | -0.470 [-0.608, -0.339] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 0.607 [0.476, 0.735] | -0.389 [-0.523, -0.263] * |

## 4. Alert simulation (cutoff = 99th percentile of benign users' day scores, per model)

| Model | detected: all | scenario 1 | scenario 2 | scenario 3 | median days to first alert | insiders alerted BEFORE window | benign users alerted |
|---|---|---|---|---|---|---|---|
| Gradient boosting (supervised), v1 (no email) | 67 of 69 | 29 of 29 | 30 of 30 | 8 of 10 | 0 | 41 of 69 | 16 of 30 |
| Gradient boosting (supervised), v2 (+ email) | 68 of 69 | 29 of 29 | 30 of 30 | 9 of 10 | 0 | 38 of 69 | 18 of 30 |
| Gradient boosting (supervised), email features only | 40 of 69 | 1 of 29 | 30 of 30 | 9 of 10 | 5 | 40 of 69 | 16 of 30 |
| GRU over the previous 14 days (supervised), v1 (no email) | 67 of 69 | 28 of 29 | 30 of 30 | 9 of 10 | 0 | 43 of 69 | 7 of 30 |
| GRU over the previous 14 days (supervised), v2 (+ email) | 65 of 69 | 28 of 29 | 29 of 30 | 8 of 10 | 0 | 47 of 69 | 4 of 30 |
| GRU over the previous 14 days (supervised), email features only | 30 of 69 | 0 of 29 | 29 of 30 | 1 of 10 | 10 | 25 of 69 | 10 of 30 |

_All users, cross-validated: development-style numbers, optimistic. Email features describe envelope data only (counts, recipients, domains, attachments, size, hour), never message text._
