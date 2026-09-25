import os

data_dir = "data/raw"
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if "answer" in f.lower() or "insider" in f.lower():
            print(os.path.join(root, f))
    for d in dirs:
        if "answer" in d.lower() or "insider" in d.lower():
            print(os.path.join(root, d) + "/ (folder)")

print("Search complete.")