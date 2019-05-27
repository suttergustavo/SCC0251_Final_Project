# SCC0251 Image Processing - Final Project

**Students:**
- Gustavo Sutter Pessurno de Carvalho  - 9763193
- Rodrigo Geurgas Zavarizz - 9791080
- Victor Henrique de Souza Rodrigues - 9791027

## Improving image upscaling quality with convolutional neural networks

### Abstract

The aim of this project is to use a machine learning approach to the image upscaling problem. Our goal is to develop a deep neural network capable of generating a higher resolution image given a low resolution sample, a process known as super resolution. Finally, once the method is implemented, we will compare our results with other approaches, such as bilinear and nearest neighbour interpolations.

### Dataset
The [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) will be used to train and test our model. Although the dataset contains imagens and attributes information only the image data will be used. There are 202,599 images with dimension 218x178x3 that have already been aligned and cropped.

To performe the super resolution task this data is used to generate two datasets: one with high resolution (HR) and the other with low resolution (LR). The LR dataset is obtained by downscaling the original images by a factor of two, while the HR dataset contains the original images.

| Example HR| Example LR |
|--|--|
| ![enter image description here](images/example_HR.png) | ![enter image description here](images/example_LR.png)|


### End-to-end super resolution neural network

As mentioned, the goal of this project is to build a convolutional neural network capable of increasing the quality of an image that is given as an input. The approach proposed is end-to-end, that is, the mapping from the LR image to the HR image is done only by the neural network. The only necessary pre-processing is to upscale the LR image to the desired size using an interpolation method. 

The following diagram demonstrates how this process works

<p align="center"> 
<img src="images/dip_flow.png">
</p>


#### Neural network architecture


#### Training the neural network



