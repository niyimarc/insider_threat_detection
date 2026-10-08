# Frozen insider detector: FINAL evaluation on the LOCKED users (`locked`)


Locked users: 49 (34 insiders by scenario {'1': 14, '2': 15, '3': 5}, 15 benign), 12406 user-days. Method fixed in FROZEN_SUPERVISED_METHOD.md (SHA-256 `0fc6ced3535f14a0...`). Commit `8365deab80145af44bca29d8235ba127ecb6dbb1`. 40 input columns (hash `cd5de52a042dd00a...`). Models trained on the 50 development users only, seeds [0, 1, 2] averaged; alert cutoffs set on development benign users' out-of-fold scores.

Read with care: only 15 benign locked users exist, so every comparison against benign users is coarse. Scenario 2 supplies most malicious days; controls are the same model given only the date or only the user's own clock. Whatever the numbers are, they are the result.

## 1. Whole-timeline person ranking (no knowledge of when anything happened)

#### rule: top_k_mean

| Model | user-level AUROC [95% CI] | minus gradient boosting [95% CI] |
|---|---|---|
| Gradient boosting (supervised) | 0.633 [0.439, 0.822] | (reference) |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 0.103 [0.009, 0.269] | -0.526 [-0.728, -0.316] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 0.103 [0.000, 0.267] | -0.527 [-0.731, -0.316] * |
| GRU over the previous 14 days (supervised) | 0.794 [0.651, 0.920] | +0.164 [+0.006, +0.335] * |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 0.420 [0.210, 0.653] | -0.208 [-0.480, +0.088] |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 0.475 [0.259, 0.709] | -0.153 [-0.440, +0.161] |

#### rule: max

| Model | user-level AUROC [95% CI] | minus gradient boosting [95% CI] |
|---|---|---|
| Gradient boosting (supervised) | 0.747 [0.574, 0.898] | (reference) |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 0.533 [0.500, 0.600] | -0.210 [-0.365, -0.041] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 0.325 [0.188, 0.482] | -0.420 [-0.616, -0.208] * |
| GRU over the previous 14 days (supervised) | 0.853 [0.710, 0.973] | +0.109 [-0.065, +0.288] |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 0.439 [0.251, 0.657] | -0.301 [-0.535, -0.050] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 0.480 [0.263, 0.714] | -0.261 [-0.527, +0.026] |

| Model | top 5 | top 10 | insiders among all locked users |
|---|---|---|---|
| Gradient boosting (supervised) | 5 of 5 | 7 of 10 | 69% |
| GRU over the previous 14 days (supervised) | 4 of 5 | 9 of 10 | 69% |

## 2. Matched-period check (insider's window vs benign users on the same dates)

| Model | matched-period AUROC [95% CI] | coverage |
|---|---|---|
| Gradient boosting (supervised) | 0.963 [0.899, 1.000] | 80% |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 0.443 [0.387, 0.483] | 80% |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 0.610 [0.526, 0.702] | 80% |
| GRU over the previous 14 days (supervised) | 0.968 [0.890, 1.000] | 80% |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 0.692 [0.575, 0.827] | 80% |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 0.729 [0.590, 0.874] | 80% |

## 3. Alert simulation (frozen cutoff = 99th percentile of benign DEVELOPMENT users' day scores)

| Model | insiders detected in their window | median days to first alert | detected within the first week | insiders alerted BEFORE their window | benign users alerted | alert days per benign user |
|---|---|---|---|---|---|---|
| Gradient boosting (supervised) | 33 of 34 (97%) | 0 | 91% of insiders | 15 of 34 | 10 of 15 | 31.1 |
| GRU over the previous 14 days (supervised) | 18 of 34 (53%) | 4 | 32% of insiders | 9 of 34 | 2 of 15 | 3.1 |

## 4. Day level: insiders only, date-stratified (window days vs the same insiders' other days, same week)

#### Insiders only, date-stratified AUROC

Window labelling. Controls should be near 0.5.

Computed on 44 of 69 calendar strata that contain both classes, covering 685 of 685 malicious rows. 0.5 = no separation.

| Model | runs | date-stratified AUROC [95% CI] | minus Gradient boosting (supervised) [95% CI] |
|---|---|---|---|
| Gradient boosting (supervised) | 1 | 0.859 [0.780, 0.926] | (reference) |
| CONTROL (calendar only (date and weekday)): Gradient boosting (supervised) | 1 | 0.503 [0.495, 0.514] | -0.357 [-0.425, -0.277] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): Gradient boosting (supervised) | 1 | 0.507 [0.489, 0.530] | -0.353 [-0.424, -0.274] * |
| GRU over the previous 14 days (supervised) | 1 | 0.877 [0.794, 0.942] | +0.018 [-0.075, +0.092] |
| CONTROL (calendar only (date and weekday)): GRU over the previous 14 days (supervised) | 1 | 0.498 [0.483, 0.510] | -0.363 [-0.431, -0.283] * |
| CONTROL (per-user clock only (days since the user's first activity, and the user's day count)): GRU over the previous 14 days (supervised) | 1 | 0.524 [0.487, 0.576] | -0.336 [-0.416, -0.251] * |

`*` = the interval for the difference excludes 0. Intervals resample whole users.

## Ranked locked users, Gradient boosting (supervised) (top 10 of 49; full list in the CSV)

`is_insider` and `scenario` come from the answer key and are for EVALUATION only.

| rank | user | user score | alert days | insider (scenario) |
|---|---|---|---|---|
| 1 | CEJ0109 | 0.988 | 73 | yes (2) |
| 2 | CCL0068 | 0.987 | 35 | yes (2) |
| 3 | NWT0098 | 0.985 | 76 | yes (2) |
| 4 | EGD0132 | 0.98 | 41 | yes (2) |
| 5 | IKR0401 | 0.98 | 60 | yes (2) |
| 6 | HXL0968 | 0.977 | 87 | yes (2) |
| 7 | PSF0133 | 0.972 | 39 | yes (2) |
| 8 | UAD0771 | 0.97 | 89 | no |
| 9 | BVC0790 | 0.967 | 173 | no |
| 10 | NDM0417 | 0.96 | 70 | no |

## Ranked locked users, GRU over the previous 14 days (supervised) (top 10 of 49; full list in the CSV)

`is_insider` and `scenario` come from the answer key and are for EVALUATION only.

| rank | user | user score | alert days | insider (scenario) |
|---|---|---|---|---|
| 1 | HXL0968 | 6.44 | 133 | yes (2) |
| 2 | MPM0220 | 5.83 | 144 | yes (3) |
| 3 | CCL0068 | 5.78 | 24 | yes (2) |
| 4 | RSC0089 | 5.69 | 46 | no |
| 5 | TNM0961 | 5.69 | 54 | yes (2) |
| 6 | IUB0565 | 5.68 | 27 | yes (2) |
| 7 | CEJ0109 | 5.56 | 12 | yes (2) |
| 8 | PSF0133 | 5.54 | 26 | yes (2) |
| 9 | LCC0819 | 5.46 | 14 | yes (2) |
| 10 | JJM0203 | 5.26 | 13 | yes (2) |

_Locked-partition numbers from the frozen method: the final, once-only evaluation. Disclosures are in FROZEN_SUPERVISED_METHOD.md._
