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

pip install -qq git+https://www.github.com/keras-team/keras-contrib.git

echo "Keras Contrib is installed"

echo "Create the Directories for Dataset"

mkdir trainA trainB valA valB testA testB

python prepare_dataset.py
echo "Dataset folder structure created"


# To delete the copied images
# !rm -rf trainA
# !rm -rf trainB

# # Move Sampel Test file
ls ./trainA/ | shuf -n 50 | xargs -i mv ./trainA/{} valA/
ls ./trainB/ | shuf -n 50 | xargs -i mv ./trainB/{} valB/

# # Move Sampel Test file
ls ./trainA/ | shuf -n 50 | xargs -i mv ./trainA/{} testA/
ls ./trainB/ | shuf -n 50 | xargs -i mv ./trainB/{} testB/

pip install -r requirements.txt
