
dataset='crop_part1'
from glob import glob
import os

import shutil
os.makedirs('trainA',exist_ok=True) # 18 to 25
os.makedirs('trainB',exist_ok=True) # 50+

os.makedirs('valA',exist_ok=True) # 18 to 25
os.makedirs('valB',exist_ok=True) # 50+
os.makedirs('testA',exist_ok=True) # 18 to 25
os.makedirs('testB',exist_ok=True) # 50+

# print(files[:4])
count_a=count_b=0
for fn in glob(dataset+'/*'):
  age=int(fn.split('/')[-1].split('_')[0])
  if age>=18 and age <=25:
    shutil.copy(fn,fn.replace('crop_part1','trainA'))
    count_a+=1
  elif age>=50 and age<=60:
    shutil.copy(fn,fn.replace('crop_part1','trainB'))
    count_b+=1
  else:
    continue

print("No of images in TrainA and TrainB Respectively are {},{}".format(count_a,count_b))
