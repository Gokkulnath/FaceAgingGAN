import glob
import random
import os



'''
We Assume the following folder structure for creating the Custom Dataset.
Dataset_NAME/
    - train/
        - A/
          - img1.jpg
          - img2.jpg
          - ...
        - B/
          - img1.jpg
          - img2.jpg
          - ...
    - test/
        - A/
          - img1.jpg
          - img2.jpg
          - ...
        - B/
          - img1.jpg
          - img2.jpg
          - ...
 '''

from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CycleGANDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned # used to handle cases when the number of images are not equal

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]) # Safe-indexing such that always withing the range

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])


        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))