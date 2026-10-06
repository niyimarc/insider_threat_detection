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
%cd /content/project
!git log --oneline -1
!ls data/raw data/processed checkpoints
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
```

Drive layout this expects (`/content/drive/MyDrive/insider_threat/`):

```
raw/r4.2/{logon,device,http,email,file}.csv  (+ psychometric.csv, LDAP/)
raw/answers/                                   (insiders.csv, r4.2-1/ ...)
processed/file_events_processed.parquet
processed/http_domain_daily.parquet, http_hour_daily.parquet
checkpoints/
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
!python scripts/precompute_http_daily.py     # once; long (reads the 14 GB http.csv); build_dataset needs its output
!python scripts/build_dataset.py             # after any feature change
!python scripts/train_models.py              # prints "Training device: cuda" on GPU
!python scripts/evaluate_models.py           # dev partition by default
```

Named runs (use these for ablations and multiple seeds, so runs never overwrite each other). Train and evaluate must get the SAME `INSIDER_CHECKPOINT_DIR`. `INSIDER_REQUIRE_GPU=1` makes training stop immediately if the runtime has no GPU (a reset runtime goes back to CPU, and a CPU run takes many hours):

```
# all signal groups, seed 1
!INSIDER_REQUIRE_GPU=1 INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 INSIDER_SEED=1 python scripts/train_models.py
!INSIDER_CHECKPOINT_DIR=checkpoints/all_seed1 python scripts/evaluate_models.py

# ablation: without http, seed 1
!INSIDER_FEATURE_GROUPS=file,device,logon INSIDER_CHECKPOINT_DIR=checkpoints/nohttp_seed1 INSIDER_SEED=1 python scripts/train_models.py
!INSIDER_FEATURE_GROUPS=file,device,logon INSIDER_CHECKPOINT_DIR=checkpoints/nohttp_seed1 python scripts/evaluate_models.py
```

Each folder gets `feature_config.json` and `run_metadata.json` (seed, versions, git commit). Evaluating with a different feature selection than the run was trained with stops with a clear error.

Diagnostics and ablations:

```
!python scripts/diagnose_scenario2_dilution.py
!python scripts/ensemble_scores.py
!INSIDER_FEATURE_GROUPS=file,device python scripts/train_models.py   # example ablation
```

The **locked** test partition is for the single final evaluation only:

```
!INSIDER_TEST_PARTITION=locked python scripts/evaluate_models.py
```

Do not use it while designing features or tuning.

Rebuilding the dataset (`build_dataset.py`) makes existing checkpoints stale. Retrain afterwards.

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
