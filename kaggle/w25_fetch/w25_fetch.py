import os, shutil, glob
print("inputs:", glob.glob("/kaggle/input/*"), glob.glob("/kaggle/input/*/*")[:40])
hits = glob.glob("/kaggle/input/**/test.csv", recursive=True)
print("test.csv hits:", hits)
for h in hits:
    if "2nd" in h:
        shutil.copy(h, "/kaggle/working/test_2025.csv"); print("copied", os.path.getsize("/kaggle/working/test_2025.csv"))
