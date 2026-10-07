# Running the project on Google Colab

## Colab basics (read first)

- A line starting with `!` runs as a shell command: `!git pull`. Without the `!`, Colab treats it as Python and raises `SyntaxError`.
- `%cd /some/folder` changes the working directory for later cells. `!cd` does not persist between cells.
- Each cell must be run with the play button (or Shift+Enter), in order.
- Everything in `/content` is **wiped when the runtime disconnects**. Only Google Drive survives. Keep data, processed files and checkpoints on Drive.

## Where things live

| What | Where |
|---|---|
| Code | GitHub (private repo), cloned into `/content/project` each session |
| Raw data, processed parquet, checkpoints | Google Drive: `/content/drive/MyDrive/insider_threat/` |
| GitHub token | Colab Secrets, name `GITHUB_TOKEN`, notebook access ON |

Never paste the token into a cell or into chat. If it is ever exposed, revoke it on GitHub and create a new one.

## 0. Choose the GPU (before anything else)

Runtime -> Change runtime type -> **T4 GPU** -> Save. Then run:

```
!nvidia-smi
```

If it prints a GPU table, you are good. If it says `command not found`, there is no GPU: Runtime -> Disconnect and delete runtime, reselect T4, and check again. Training on CPU works but is far slower.

## Quick setup: ONE cell for every new session

Colab runtimes are temporary and get recycled (idle timeout, time limit, closing the browser). Data on Drive is never lost, so recovery is just this cell. It is safe to run at any time: it mounts Drive, clones the code (or pulls if it is already there), installs requirements, and re-links the Drive folders.

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
token = userdata.get('GITHUB_TOKEN')
D = "/content/drive/MyDrive/insider_threat"

!mkdir -p {D}/raw {D}/processed {D}/checkpoints
!if [ -d /content/project/.git ]; then git -C /content/project pull; else git clone https://{token}@github.com/niyimarc/insider_threat_detection.git /content/project; fi
!pip install -q -r /content/project/requirements.txt

!mkdir -p /content/project/data
!rm -rf /content/project/checkpoints
!ln -sfn {D}/raw /content/project/data/raw
!ln -sfn {D}/processed /content/project/data/processed
!ln -sfn {D}/checkpoints /content/project/checkpoints
# results/ also lives on Drive, so the results tables survive a runtime reset
!mkdir -p {D}/results
!if [ ! -L /content/project/results ]; then rm -rf /content/project/results; fi
!ln -sfn {D}/results /content/project/results
%cd /content/project
!git log --oneline -1
!ls data/raw data/processed checkpoints results
```

Then check the GPU (`!nvidia-smi`) and run whatever step you need. The expected `ls` output: `raw/` holds `r4.2` and `answers`, `processed/` holds the parquet files, `checkpoints/` holds the model files. If `r4.2` is missing, do the one-off raw-data download in section 1.

What cannot be made permanent: a running job. If the runtime is recycled mid-run, that run is lost and restarts from the beginning (training and the http precompute have no resume).

## 1. Full setup, step by step (same thing as the quick setup, in separate cells)

Run these cells in order.

**Cell 1 - mount Drive** (tick every box in the popup; allow popups)

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 - clone the code**

```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/niyimarc/insider_threat_detection.git /content/project
```

**Cell 3 - install requirements**

```
%cd /content/project
!pip install -q -r requirements.txt
```

**Cell 4 - link the Drive folders into the project** (so nothing is copied and results persist)

```
!mkdir -p /content/project/data
!rm -rf /content/project/checkpoints
!ln -sfn /content/drive/MyDrive/insider_threat/raw /content/project/data/raw
!ln -sfn /content/drive/MyDrive/insider_threat/processed /content/project/data/processed
!ln -sfn /content/drive/MyDrive/insider_threat/checkpoints /content/project/checkpoints
!mkdir -p /content/drive/MyDrive/insider_threat/results
!if [ ! -L /content/project/results ]; then rm -rf /content/project/results; fi
!ln -sfn /content/drive/MyDrive/insider_threat/results /content/project/results
```

Drive layout this expects (`/content/drive/MyDrive/insider_threat/`):

```
raw/r4.2/{logon,device,http,email,file}.csv  (+ psychometric.csv, LDAP/)
raw/answers/                                   (insiders.csv, r4.2-1/ ...)
processed/file_events_processed.parquet
processed/http_domain_daily.parquet, http_hour_daily.parquet
checkpoints/
results/unsupervised/                          (final_evaluation_*.json / .md tables)
```

**One-off: download the raw data to Drive** (only if `raw/r4.2` is missing; about 4.8 GB download, 10-25 min to extract, keep the tab open)

```
%%bash
set -e
cd /content
wget -q --show-progress -O r4.2.tar.bz2 https://ndownloader.figshare.com/files/24856766
echo "cf64caa378acb77cd0c608a5576d998c  r4.2.tar.bz2" | md5sum -c
tar -xjf r4.2.tar.bz2 -C /content/drive/MyDrive/insider_threat/raw
rm r4.2.tar.bz2
ls -la /content/drive/MyDrive/insider_threat/raw/r4.2
```

**Cell 5 - check it worked**

```
!ls /content/project/data/raw /content/project/data/processed /content/project/checkpoints
!git -C /content/project log --oneline -1
```

## 2. Updating code after you push to GitHub

If the session is still alive (`/content/project` exists):

```
%cd /content/project
!git pull
```

If `git pull` says `not a git repository` or `/content/project` does not exist, the session was reset: do the full setup in section 1.

If the clone says `destination path already exists`, the folder is already there: use `%cd /content/project` and `!git pull` instead.

## 3. Running the pipeline

Always from the project root (`%cd /content/project`).

```
!python scripts/common/precompute_http_daily.py     # once; long (reads the 14 GB http.csv); build_dataset needs its output
!python scripts/common/build_dataset.py             # after any feature change
!python scripts/unsupervised/train_models.py              # prints "Training device: cuda" on GPU
!python scripts/unsupervised/evaluate_models.py           # dev partition by default
```

Named runs (use these for ablations and multiple seeds, so runs never overwrite each other). Train and evaluate must get the SAME `INSIDER_CHECKPOINT_DIR`. `INSIDER_REQUIRE_GPU=1` makes training stop immediately if the runtime has no GPU (a reset runtime goes back to CPU, and a CPU run takes many hours):

```
# all signal groups, seed 1
!INSIDER_REQUIRE_GPU=1 INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 INSIDER_SEED=1 python scripts/unsupervised/train_models.py
!INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 python scripts/unsupervised/evaluate_models.py

# ablation: without http, seed 1
!INSIDER_FEATURE_GROUPS=file,device,logon INSIDER_CHECKPOINT_DIR=checkpoints/nohttp_seed1 INSIDER_SEED=1 python scripts/unsupervised/train_models.py
!INSIDER_FEATURE_GROUPS=file,device,logon INSIDER_CHECKPOINT_DIR=checkpoints/nohttp_seed1 python scripts/unsupervised/evaluate_models.py
```

Each folder gets `feature_config.json` and `run_metadata.json` (seed, versions, git commit). Evaluating with a different feature selection than the run was trained with stops with a clear error.

**Models** are declared once in `src/unsupervised/models/registry.py` (the two autoencoders, the Isolation Forest, and two heuristic baselines that need no training: the single http feature and session length). Training, evaluation and the ensemble loop over that list. `INSIDER_MODELS=lstm_ae,static_iforest` restricts any script to a subset.

**The results table** (`scripts/unsupervised/final_evaluation.py`): AUROC/AUPRC averaged over the seeds you pass, with 95% user-level bootstrap intervals, a paired comparison against the single-feature baseline, and BOTH labelings (window and answer-key) side by side. It writes `results/unsupervised/final_evaluation_<partition>_<tag>.json` and `.md`.

```
# development partition (the default), three seeds of the same configuration
!python scripts/unsupervised/final_evaluation.py --runs checkpoints/all_seed1 checkpoints/all_seed2 checkpoints/all_seed3 --tag all

# the same for an ablation (the INSIDER_FEATURE_GROUPS value must match how those runs were trained)
!INSIDER_FEATURE_GROUPS=file,device,logon python scripts/unsupervised/final_evaluation.py --runs checkpoints/nohttp_seed1 checkpoints/nohttp_seed2 --tag nohttp
```

The LOCKED partition is for one final evaluation of a method fixed in advance, and needs `--partition locked --confirm-final`. Whatever it shows is what gets reported. Run it only once the method, feature set and seeds are frozen.

Diagnostics and ablations:

```
!python scripts/unsupervised/diagnose_scenario2_dilution.py
!python scripts/unsupervised/ensemble_scores.py
!INSIDER_FEATURE_GROUPS=file,device python scripts/unsupervised/train_models.py   # example ablation
```

The **locked** test partition is for the single final evaluation only:

```
!INSIDER_TEST_PARTITION=locked python scripts/unsupervised/evaluate_models.py
```

Do not use it while designing features or tuning.

Rebuilding the dataset (`build_dataset.py`) makes existing checkpoints stale. Retrain afterwards.

## 3b. Supervised user-day detection (no GPU needed)

Trains logistic regression and gradient boosting on user-day features with the WINDOW label, cross-validated over USERS (stratified by scenario), on the development partition only. It needs the processed dataset (`build_dataset.py`) and nothing else, runs on the CPU in a few minutes, and writes `results/supervised/supervised_cv_dev_<tag>.json` and `.md`.

```
!python scripts/supervised/cross_validate.py
# a quicker check of two configurations:
!python scripts/supervised/cross_validate.py --models gradient_boosting --configs file all --tag quick
```

It reports four signal-group configurations (file; +device; +logon; +http), both labellings (window = training label; answer key = evaluation only), two no-training heuristics, and the http signal's paired contribution. There is deliberately no locked-partition option yet.

It ends with a DATE-MATCHED EVALUATION, which is the number to quote: the date-stratified AUROC compares each malicious day only with non-malicious days from the same calendar week (so knowing the date cannot help), reported for normally trained models and for models retrained with date-matched weights. The calendar-only control must land near 0.5 there; if it does not, the correction is not working. `--strata-days` changes the stratum width (default 7).

The last section is the DECISIVE CHECK: insiders only, date-stratified. Only insiders' days are scored, each malicious day against other insiders' non-malicious days from the same week, so neither the date nor "insiders differ from benign users" can explain a score; the third table also trains on insiders only. A model clearly above 0.5 there, with the calendar-only control near chance, is separating behaviour inside the window from behaviour outside it. If a week range has no insiders on both sides of their window, the section says so instead of reporting a number.

It also runs three CONFOUND CHECKS, to be read before any of the scores: a calendar-only control (the same models given only the date; a high score means the window labels can be recovered from when a day happened, not from behaviour), a within-insider comparison (insiders' window days vs the same insiders' other days, paired against that control), and permutation importance by signal group.

## 3c. Supervised detection on ALL user-days (device and logon signals)

The table above only had days with a file copy. This version has a row for EVERY day the user was active in the logon, device or file log, and adds day-level device and logon signals (connects per day, after-hours events, distinct PCs, first-ever drive use, and deviations from the user's own previous days; see `src/supervised/day_signals.py`). The feature list was fixed in advance from the dataset's readme and `scenarios.txt`, not from results. It needs the raw `logon.csv` and `device.csv` on Drive as well as the processed dataset, and it runs on the CPU. The table is about five times larger than before, so expect a long run; a quick first pass:

```
!python scripts/supervised/cross_validate_all_days.py --models gradient_boosting --n-repeats 1 --n-boot 200 --tag quick
```

and the full run (all models, 3 repeats, 500 resamples):

```
!python scripts/supervised/cross_validate_all_days.py
```

Ablation (remove input columns and rerun, e.g. the cumulative device-use counter):

```
!python scripts/supervised/cross_validate_all_days.py --exclude-features dv_prior_use_days --tag no_prior_use
```

It writes `results/supervised/supervised_all_days_dev_<tag>.json` and `.md` with the same sections as 3b, plus "Permutation importance by individual feature" (top 15 columns, showing which device/logon feature carries the group-level result). Read the last section (insiders only, date-stratified) first. Dev only: there is deliberately no locked-partition option.

## 4. When Colab disconnects

Disconnects happen (idle timeout, browser closed, usage limits). What survives and what does not:

- Survives: everything on Drive (raw data, processed parquet, checkpoints written there).
- Lost: the cloned code, installed packages, any run in progress, and anything in `/content`.

To recover:

1. Reconnect (Runtime -> Connect), re-check the GPU (`!nvidia-smi`).
2. Run the full setup, section 1.
3. Re-run whichever step was interrupted. Training and the http precompute have **no resume**, so an interrupted run starts again from its beginning.

Tips to reduce disconnects:

- Keep the browser tab open while a long job runs.
- Run long jobs with the runtime idle timeout in mind; do not start a multi-hour job right before leaving unless you can check on it.
- Long jobs write their outputs to Drive at the end, so check the files exist (`!ls -la /content/project/checkpoints`) before assuming a run finished.

## 5. Common errors

| Error | Fix |
|---|---|
| `SyntaxError` on `git pull` / `ls` / `python ...` | Add `!` at the start of the line |
| `credential propagation was unsuccessful` | Tick all boxes in the auth popup, allow popups, or use the Files sidebar -> Mount Drive button |
| `SecretNotFoundError` | Create the `GITHUB_TOKEN` secret and enable Notebook access for it |
| `could not read Username` | Use the `https://{token}@github.com/...` clone URL (Cell 2) |
| `not a git repository` / `/content/project` missing | Session reset: run section 1 again |
| `destination path already exists` | `%cd /content/project` then `!git pull` |
| `nvidia-smi: command not found` | No GPU: disconnect and delete runtime, reselect T4 |
| `size mismatch` loading a checkpoint | Checkpoints are from an older feature set or model size: retrain |
| `no 'test_partition' column` | Dataset predates the dev/locked split: run `build_dataset.py` |
