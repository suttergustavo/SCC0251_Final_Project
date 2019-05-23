
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

To performe de super resolution task this data is use to generate two dataset one with high resolution (HR) and the other with low resolution (LR). The LR dataset is obtained by downscaling the original images by a factor of two, while the HR dataset contains the original images.

| Exemple HR| Exemple LR |
|--|--|
| ![enter image description here]() | ![enter image description here] ()|



