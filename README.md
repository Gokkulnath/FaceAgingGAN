# FaceAgingGAN



About the Dataset : 
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. Some sample images are shown as following

![Sample Image](resources/samples.png)

Split the Dataset into 2 Folders :

Age greater than  15 and less than 30 Years --> Young
Age greater than 50 Years --> old (Try age range of 40-50 Years)
We use this dataset to train a Cycle GAN

# Cycle GAN Architecture
TLDR: Learn Mapping functions (Generative models) between two domains X and Y (Not necessarly paired) using GAN Architecture.


![](resources/cyclegan_arch.png)
We Create 2 GAN with the objective to generate younger-older equivalents of old-young input images.


## Loss Objectives : 

Adversarial Loss :  Matching the Distribution of Generated Images to data distribution in the target domain. Similar to Normal GAN Loss, the idea is to use a discriminator network to classify the generated images as Fake or Real. In CycleGAN Case we have two discriminators which will evaluate the performance of Generators ability to model the target distribution.



Cycle Consistency Loss : Prevents the learned mappings of X and Y from contradicting each other.

Adversarial Loss alone can map the same set of input images to any random permutation of images in the target domain, where any of the 


Identity Loss (Optional) : Use only when we need to preserved the color composition between input and output(eg. Paintings to Photos)
It is achieved by Regularizing the generator near an idenity mapping when real images are fed into the generator.
Prevents Altering of Tint of Input images

# TODO Need to Add Clarity



## Steps to Run the Experiment Locally:

1. Run the **setup_experiment.bash** --> Downloads the dataset and creates the required folder structure
2. Run python run.py
3. Results are stored in images directory
4. Trained Model weights can be obtained from saved_models directory


### Hyperparameters:


Youtube Video: [Link](https://youtu.be/adP4hCEpPHU)


[![Results](https://img.youtube.com/vi/adP4hCEpPHU/0.jpg)](https://www.youtube.com/watch?v=adP4hCEpPHU)


- Notable Changes:
  -  Old --> Young : The Network has learned to remove wrinkles and add a lit bit of fairness to the face.
  -  Young --> Old : Tries to add wrinkles with a noisy patch pattern (Not perfect) and transforms eyebrow with appropriate aging effects.

The Results might not be that great because of  choice of age range and the large variance in the old class age. We can add/get more data and train for longer iterations/epochs to obtain better perfromance. 


## Limitations of CycleGAN 

- Tasks Which required Geometric Changes. (Can lead to Transfiguration)



## Misc Notes :

Style Transfer vs CycleGAN : 
Style Transfer is setup to transfer styles from a single style image/instance and optimize accordingly, while CycleGAN can generate Stylized imgaes for multiple style images/instances


Horse to Zerbra: Humans also getting striped ?
During trainning the model did not encounter images which have humans riding the horse or zebra

