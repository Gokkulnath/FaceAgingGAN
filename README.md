# FaceAgingGAN



About the Dataset : 
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. Some sample images are shown as following

![Sample Image](resources/samples.png)

Split the Dataset into 2 Folders :

Age less than 30 Years  and 
Age greater than 50 Years



We use this dataset to train a Cycle GAN

Generate 2 GAN with the objective to generate younger-older equivalents of old-young input images.


Cycle GAN Architecture


Steps to Run the Experiment Locally:

1. Run the **setup_experiment.bash** --> Downloads the dataset and creates the required folder structure
2. Run python main.py
3. Results are stored in images directory


