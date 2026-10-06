# Data Exploration Log

This folder holds the scripts used to understand the raw CERT r4.2 dataset
and its answer key before any preprocessing or modelling code was written.
Each entry below records what the script does and, where relevant, what it
found — several of these findings directly shaped decisions in Chapter 3.

## 01_explore_raw_data.py
Walks the full folder structure under `data/raw/` and prints it as an
indented tree. Used to confirm exactly what the extracted CERT r4.2 archive
contains before writing any code that assumes a particular layout, rather
than guessing folder/file names in advance.

## 02_find_answer_key.py
Searches recursively under `data/raw/` for any file or folder with "answer"
or "insider" in its name. Finding: nothing was found in the initially
extracted `r4.2.tar.bz2` archive — the ground-truth labels are not bundled
with the activity logs. Confirmed (by checking the original figshare
listing) that the answer key is a separate download, `answers.tar.bz2`,
hosted on the same dataset page.

## 03_inspect_answer_key.py
After extracting `answers.tar.bz2`, this reads the answer key's own
`readme.txt` and `scenarios.txt`, and previews the first 10 lines of one
per-user incident file (`r4.2-1-AAM0658.csv`). Finding: the answer key
documents five possible red-team scenarios in total; r4.2 only implements
three of them (folders `r4.2-1`, `r4.2-2`, `r4.2-3`). The sample file's
contents (an after-hours logon, a device connect, then an HTTP request to
wikileaks.org) matched the scenario 1 description exactly, confirming the
answer key is a script of malicious behaviour written in advance by the
dataset creators, not a record of after-the-fact investigation.

## 04_inspect_insiders_csv.py
Loads `answers/insiders.csv` — described in the answer key's readme as
"the master file of true positives" — and prints its shape, column names,
and first 10 rows. Finding: 191 total rows (one per malicious incident,
across every CERT release combined), with columns `dataset, scenario,
details, user, start, end`.

## 05_filter_r42_labels.py
Filters `insiders.csv` down to just `dataset == 4.2` and prints the result.
Finding: exactly 70 matching rows (30 for scenario 1, 30 for scenario 2, 10
for scenario 3), independently confirming the 70-malicious-user figure
already used in Chapters 2 and 3, and matching the per-scenario folder
counts found in script 03.

## 06_inspect_file_csv.py
Checks the on-disk size of each main log file and previews the first 5 rows
of `file.csv`. Finding: `http.csv` (13.9 GB) and `email.csv` (1.3 GB) are
far too large to be practical, reinforcing the existing decision to exclude
them from this project's scope. `file.csv`'s actual columns are `id, date,
user, pc, filename, content` — notably, no column recording the type of
file operation (open/write/copy/delete), which Chapter 3.4 originally
assumed would be available.

## 07_verify_file_csv_schema.py
Before concluding anything from a 5-row preview, this reads r4.2's own
`readme.txt` (the dataset's authoritative documentation, which had not yet
been checked), loads a 50,000-row sample of `file.csv`, and searches the
`content` field for the words "open", "write", "copy", "delete" in case
operation type was embedded in the text rather than a separate column.
Finding: r4.2's readme states plainly that "each entry represents a file
copy to a removable media device" — there is only one operation type in
this file, which is why no such column exists. The keyword search in
`content` returned only coincidental matches (content is randomly generated
topic text, not a description of the action), ruling out the alternative
explanation.

## 08_full_schema_verification.py
Loads `file.csv`, `logon.csv`, and `device.csv` in full (not sampled) to
verify the schema holds across the entire file, not just an early slice.
Also extracts the hexadecimal file-header from the start of each `content`
value and cross-tabulates it against filename extension. Findings:
`logon.csv` and `device.csv` match their documented format exactly
(Logon/Logoff and Connect/Disconnect, nothing hidden). `file.csv` contains
445,581 rows from only 264 distinct users (not all 1,000). The file-header
values are dominated by five common types (matching doc/pdf/zip/jpg/exe)
covering ~95% of rows, with a long tail of ~22,600 rare header values to be
bucketed as "other" during feature engineering. A small number of rows show
a mismatch between filename extension and actual file-header signature
(e.g. a `.jpg` file with a Word-document header), which may be worth
investigating later as a possible behavioural signal. This script also
first surfaced that `date` was being read as plain text, not a real date.

## 09_check_malicious_user_coverage.py
Cross-references the 70 malicious users (from `insiders.csv`) against the
264 users who actually appear in `file.csv`, and fixes the date-parsing bug
found in script 08 using `pd.to_datetime`. Finding: 69 of the 70 malicious
users appear in `file.csv`; only one (`KPC0073`, scenario 1) has zero
`file.csv` activity at all. Corrected date range for `file.csv`: 2010-01-02
to 2011-05-16, consistent with the ~17-month simulation period already
described in Chapters 2 and 3.

## 10_check_scenario_file_relevance.py
For all 70 malicious users, counts how many of their `file.csv` events fall
specifically inside their labelled malicious time window (`start`/`end`
from `insiders.csv`), not just whether they appear in the file at all.
Finding, broken down by scenario:
- Scenario 2 (thumb-drive theft): every user shows strong, consistent
  in-window activity (minimum 71 events, most 100+). file.csv reliably
  captures this scenario.
- Scenario 3 (disgruntled sysadmin): most users show a small but non-zero
  in-window count (often exactly 1 event), consistent with this being a
  single, sharp act rather than sustained activity. file.csv captures this
  scenario, but with a sparser signal.
- Scenario 1 (wikileaks upload): median in-window count across 30 users is
  **zero**. The malicious act here is an HTTP upload, not a file copy, so
  it is largely invisible to file.csv — any file.csv activity these users
  do have is unrelated background behaviour, not evidence of the scripted
  incident.

## 11_check_logon_patterns.py
Checks whether logon.csv actually shows the "repeated Logon with no
intervening Logoff" pattern implied by its own readme (screen locks
are not recorded, only unlocks are, and unlocks are logged as another
Logon). Finding: confirmed — 103,052 occurrences across 433 of the
roughly 1,000 users. This directly justifies the session-building logic
in src/common/preprocessing/sessionize.py, which tracks open/closed session
state per user rather than naively pairing each Logon with the next
Logoff chronologically.

## 12_verify_split_on_real_data.py
Confirms the train/validation/test split behaves correctly on the real,
full-scale data (445,581 events, 264 distinct users), not just on small
synthetic test cases. Finding: user counts match the 70/15/15 design
exactly (136/29/99, with the 30 remaining benign users and all 69
malicious users combined into test), and no user's events are split
across more than one group. The uneven event-count totals seen in the
full pipeline run are fully explained by natural per-user activity
variance, not a flaw in the split logic.

## 13_verify_dataset_on_real_data.py
Smoke-tests `src/unsupervised/training/dataset.py` (SessionDataset, collate_sessions)
and `src/unsupervised/training/vocab.py` against the real processed parquet, not just
the hand-built toy DataFrames used in the unit tests. Builds the
file_type/file_extension vocabularies from the train split only, then
reuses them for validation/test, and pulls one real batch through a
DataLoader. Findings: 41,948/7,852/14,171 train/validation/test sessions,
7 entries in each vocab (6 real categories + the reserved `<unknown>`
slot), session lengths ranging 1-35 events. Train and validation both
show a 0.0000 malicious rate — every malicious session lands in test, by
design of `split_users` — confirming the model will train on benign
sessions only and must be evaluated as an anomaly detector rather than a
standard supervised classifier. No NaNs in the collated numeric features.