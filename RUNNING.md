## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing.

### Prerequisites

You have two options to run this program first is to run the notebooks, you could use Jupyter or [Colab](https://colab.research.google.com/) from Google the other option is to run the raw scripts. For Colab you won't need to download any prerquisite. For the other options you will first need Python and [CUDA](https://developer.nvidia.com/cuda-10.0-download-archive).

To install via terminal
```
sudo apt-get install python3
```
you can also download it [here](https://www.python.org/downloads/)

Then you will need these packages to run everything
```
# numpy
sudo pip3 install numpy
# kaggle
sudo pip3 install kaggle
# pytorch
sudo pip3 install torch
# torchvision
sudo pip3 install torchvision
# matplotlib
sudo pip3 install matplotlib
# scikit-image
sudo pip3 install scikit-image
# PIL
sudo pip3 install pillow
```

At last you will need Jupyter if you will run the notebooks.
```
sudo pip3 install jupyter
```

### Installing

Download the repository from Github and you are good to go.

## Running the tests

Open the notebook [conv_net_approach.ipynb](notebooks/conv_net_approach.ipynb) on your environment or run the script [train.py](src/train.py) to train the model.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
