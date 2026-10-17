# Running the project on Google Colab

Three things to know:

- A line starting with `!` is a shell command (`!git pull`). Without the `!`, Colab treats it as Python and raises `SyntaxError`.
- `%cd` changes the folder for later cells; `!cd` does not.
- `/content` is wiped when the runtime disconnects. Only Google Drive (`/content/drive/MyDrive/insider_threat/`) survives, so raw data, processed files, checkpoints and `results/` live there. The GitHub token lives in Colab Secrets as `GITHUB_TOKEN`; never paste it into a cell or into chat.

## A. New or expired session (run first)

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

Expected: `data/raw` has `r4.2` and `answers`; `data/processed` has the parquet files. If `r4.2` is missing, download it once:

```
%%bash
set -e
cd /content
wget -q --show-progress -O r4.2.tar.bz2 https://ndownloader.figshare.com/files/24856766
echo "cf64caa378acb77cd0c608a5576d998c  r4.2.tar.bz2" | md5sum -c
tar -xjf r4.2.tar.bz2 -C /content/drive/MyDrive/insider_threat/raw
rm r4.2.tar.bz2
```

## B. After pushing new code (session still running)

```
%cd /content/project
!git pull
```

If it says `not a git repository`, the session was reset: run section A.

## C. Running the project

Always from `/content/project`. Run only the steps you need, in this order. All steps run on the CPU except step 5.

**1. Prepare the data (once; repeat `build_dataset` after any feature change)**

```
!python scripts/common/precompute_http_daily.py     # long: reads the 14 GB http.csv
!python scripts/common/precompute_email_daily.py
!python scripts/common/build_dataset.py
```

**2. Development experiments (50 development users, cross-validated)**

```
!python scripts/supervised/cross_validate_all_days.py --models gradient_boosting gru --tag dev
!python scripts/supervised/detect_insiders.py --models gradient_boosting gru --tag dev
```

Add `--n-repeats 1` to either for a faster check.

**3. Final test on the locked users (once only; the method is fixed in `FROZEN_SUPERVISED_METHOD.md`, do not edit it)**

```
!python scripts/supervised/final_detector_locked.py --confirm-final
```

**4. False alarms on the 165 further benign users**

Needs the locked-run files in `results/supervised/` (`final_detector_locked_locked.json` and the two `final_detector_users_locked_*.csv`). Check with `!ls results/supervised`; if they are missing, upload them from the repository's `results/supervised/` folder on your PC to Drive (`insider_threat/results/supervised/`).

```
!python scripts/supervised/extra_benign_false_alarms.py
```

**5. Extensions (all 99 users, cross-validated; not clean evidence)**

```
!python scripts/supervised/evaluate_email_signal.py
!python scripts/supervised/evaluate_v3_detector.py
```

**6. Preliminary unsupervised models (optional; needs a GPU: Runtime -> Change runtime type -> T4)**

```
!python scripts/unsupervised/train_models.py
!python scripts/unsupervised/evaluate_models.py
```

## D. Where the results go

Every script writes to `results/` on Drive (`results/supervised/`, `results/unsupervised/`), so they survive a disconnect. To keep them in the repository: download them from Drive into the same folder on your PC, then

```
git add -f results/supervised/<file>
git commit -m "Add <what>"
git push
```

`-f` is needed because `results/` is ignored by git.

| Step | Main output files |
|---|---|
| 2 | `supervised_all_days_dev_<tag>.md/.json`, `detector_dev_<tag>.md/.json`, `detector_users_*.csv` |
| 3 | `final_detector_locked_locked.md/.json`, `final_detector_users_locked_<model>.csv`, `LOCKED_RUN_RECORD.json` |
| 4 | `extra_benign_false_alarms_extra_benign.md/.json`, `extra_benign_users_extra_benign_<model>.csv` |
| 5 | `email_signal_cv_all.md/.json`, `v3_detector_cv_all.md/.json` |

## E. If something goes wrong

A running job is lost if the runtime is recycled; rerun section A, then the step. Training and the http precompute have no resume.

| Error | Fix |
|---|---|
| `SyntaxError` on `git`, `ls` or `python` | Add `!` at the start of the line |
| `not a git repository` or `/content/project` missing | Session reset: run section A |
| `SecretNotFoundError` | Create the `GITHUB_TOKEN` secret and turn on Notebook access |
| `credential propagation was unsuccessful` | Tick all boxes in the Drive popup, or use Files sidebar -> Mount Drive |
| `no 'test_partition' column` | Run `build_dataset.py` |
| `size mismatch` loading a checkpoint | Checkpoints are from an older feature set: retrain |
| `refusing to run` on the locked step | The locked test was already run; this is intended (see `LOCKED_RUN_RECORD.json`) |
