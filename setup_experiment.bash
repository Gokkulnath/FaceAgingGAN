## Code to Download the UTKFace dataset and organize it into required folder Structure
## WARN: This script will install modules in requirements.txt in the current python environment

wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
chmod +x gdown.pl
echo "gdown.pl Dowloading done"
# Dataset not required for this Project
#./gdown.pl https://drive.google.com/open?id=0BxYys69jI14kYVM3aVhKS1VhRUk UTKFace.tar.gz
#echo "UTKFace.tar.gz Dowloading done"
#tar -zxf   UTKFace.tar.gz
#echo "UTKFace.tar.gz Extraction done"


./gdown.pl https://drive.google.com/open?id=0BxYys69jI14kRjNmM0gyVWM2bHM crop_part1.tar.gz
echo "crop_part1.tar.gz Dowloading done"
  
tar -zxf   crop_part1.tar.gz
echo "crop_part1.tar.gz Extraction done"

echo "Create the Directories for Dataset"

mkdir young old

python prepare_dataset.py
echo "Dataset folder structure created"

