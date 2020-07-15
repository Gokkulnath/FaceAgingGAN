import re
import shutil
import os
from collections import Counter
from glob import glob
import random

dataset = "crop_part1"
test_size =100

imgs =  glob(f'{dataset}/*')
age_ls=[]
gender_ls=[]
for img_path in imgs:
    age, gender = re.findall(pattern=r'(\d+)_(\d)_',string=img_path)[0]
    age_ls.append(int(age))
    gender_ls.append(int(gender))
    
old =  [age for age in age_ls if age >= 50]
young = [age for age in age_ls if age >= 15 and age <=30]
print("Number of Young : "len(young),"Number of old",len(old))

os.makedirs("train/A", exist_ok=True)
os.makedirs("train/B", exist_ok=True)
os.makedirs('test/A', exist_ok=True)
os.makedirs('test/B', exist_ok=True)

count_a = count_b = 0
for fn in glob("crop_part1/*"):
    age, _ = re.findall(pattern=r"(\d+)_(\d)_", string=fn)[0]
    age = int(age)
    if age >= 15 and age <= 30:
        shutil.copy(fn, fn.replace("crop_part1", "train/A"))
        count_a += 1
    elif age >= 50:
        shutil.copy(fn, fn.replace("crop_part1", "train/B"))
        count_b += 1
    else:
        continue
print("No of images in young is {} and old is {}".format(count_a, count_b))


## Setting up the test dataset for evaluating the performance

test_size =100

os.makedirs('test/A', exist_ok=True)
os.makedirs('test/B', exist_ok=True)
for split in ['train/A','train/B']:
    pattern = f'{split}/*'
    files = glob(pattern)
    paths = random.sample(files,k=test_size)
    for path in paths:
        shutil.move(path, path.replace("train", "test"))