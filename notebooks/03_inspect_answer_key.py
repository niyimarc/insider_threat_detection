import os

script_dir = os.path.dirname(os.path.abspath(__file__))
answers_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "answers"))

# Read the documentation files first
for doc in ["readme.txt", "scenarios.txt"]:
    path = os.path.join(answers_dir, doc)
    if os.path.exists(path):
        print(f"--- {doc} ---")
        with open(path, "r", errors="replace") as f:
            print(f.read())
        print()

# Peek at one r4.2 scenario-1 file to see its actual columns
sample_dir = os.path.join(answers_dir, "r4.2-1")
sample_file = os.listdir(sample_dir)[0]
print(f"--- sample file: {sample_file} ---")
with open(os.path.join(sample_dir, sample_file), "r", errors="replace") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 9:
            break