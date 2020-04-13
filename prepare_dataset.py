import re
import shutil
import os
from glob import glob

dataset = "crop_part1"

os.makedirs("young", exist_ok=True)
os.makedirs("old", exist_ok=True)


count_a = count_b = 0
for fn in glob("crop_part1/*"):
    age, _ = re.findall(pattern=r"(\d+)_(\d)_", string=fn)[0]
    age = int(age)
    if age >= 15 and age <= 30:
        shutil.copy(fn, fn.replace("crop_part1", "young"))
        count_a += 1
    elif age >= 50:
        shutil.copy(fn, fn.replace("crop_part1", "old"))
        count_b += 1
    else:
        continue
print("No of images in young is {} and old is {}".format(count_a, count_b))
