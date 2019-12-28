# DCGAN

The aim of the project is to generate fake human faces using deep convolutional GAN.


## Fake Images
![Image of variations](https://raw.githubusercontent.com/code-asc/DCGAN/master/Figure_1.jpeg " ")

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install cv2
pip install numpy
pip install matplotlib
pip install torch
pip install torchvision
```

## Instructions.

Download the celebs dataset and paste it in data folder
```bash 
Project----
       main.py
       generator.py
       discriminator.py
       samples.py
       data----
           celebs----
                 images
```

Run the main.py to train both discriminator and generator. It takes a while to complete execution.
Finally run samples.py file to generate the fake samples.
