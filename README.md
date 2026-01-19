# CNN Activation Functions Comparison

This repository is designed to compare the performance of S2LU and LoCLU activation functions with various CNN models. This study is based on the framework described in S. R. Dubey, S. K. Singh, and B. B. Chaudhuri's "Activation functions in deep learning: A comprehensive survey and benchmark".

## 📁 Repository Structure
├── models/
│   ├── init.py             # Models package init
│   ├── resnet50_custom.py      # ResNet50 model with custom activations
│   ├── senet18_custom.py       # SENet18 model with custom activations
│   ├── googlenet_custom.py     # GoogLeNet model with custom activations
│   ├── vgg16_custom.py         # VGG16 model with custom activations
│   ├── densenet121_custom.py   # DenseNet121 model with custom activations
│   └── mobilenet_custom.py     # MobileNet model with custom activations
├── activations/
│   ├── init.py             # Activation factory
│   ├── standard_activations.py # Standard activation functions
│   └── custom_activations.py   # S2LU and LoCLU implementations
├── datasets/
│   ├── cifar10_loader.py       # CIFAR-10 dataset loader
│   └── cifar100_loader.py      # CIFAR-100 dataset loader
├── results/                    # Analysis results and plots
├── data/                       # CIFAR datasets (auto-downloaded)
├── train_and_eval.py           # CNN model performance comparison
├── gradient_stability_analysis.py # Deep network gradient stability analysis
├── requirements.txt            # Python dependencies
└── README.md

##  Analysis Types

This repository offers two different types of analysis:

### 1. CNN Model Performance Comparison
- **File**: `train_and_eval.py`
- **Purpose**: Performance comparison of activation functions across different CNN architectures
- **Metrics**: Accuracy, Loss
- **Framework**: PyTorch
- **Source**: Developed based on S. R. Dubey et al. framework

### 2. Deep Networks Numerical Stability and Robustness Analysis
- **File**: `gradient_stability_analysis.py`
- **Purpose**: Gradient stability analysis of activation functions in very deep networks
- **Metrics**: Gradient norms, numerical stability
- **Framework**: PyTorch
- **Author**: This analysis methodology and implementation were developed by our research team

##  Activation Functions

### Standard Activation Functions
- **ReLU**: Rectified Linear Unit
- **Leaky ReLU**: Leaky Rectified Linear Unit
- **ELU**: Exponential Linear Unit
- **GELU**: Gaussian Error Linear Unit
- **Mish**: x * tanh(softplus(x))
- **Swish**: x * sigmoid(x)

### Custom Activation Functions
- **S2LU (Self-Stabilized Linear Unit)**:

S2LU(x) = ((1 + (x + α) / √(β + x²)) / 2) * x
α = 0.0025, β = 5.0

- **LoCLU (Custom Hyperbolic Linear Unit)**:

LoCLU(x) = α * x * log(1.5 + atan(x) / π)
α = 1.444

##  CNN Models

- **ResNet50**: Residual Network with 50 layers
- **SENet18**: Squeeze-and-Excitation Network with 18 layers
- **GoogLeNet**: Inception-based architecture
- **VGG16**: Visual Geometry Group 16-layer network
- **DenseNet121**: Densely Connected Convolutional Network
- **MobileNet**: Efficient architecture for mobile devices

##  Datasets

- **CIFAR-10**: 10 classes, 32x32 color images (Batch Size: 128)
- **CIFAR-100**: 100 classes, 32x32 color images (Batch Size: 64)

##  Installation

```bash
# Clone the repository
git clone <repository-url>
cd activation-functions-comparison

# Install required packages
pip install -r requirements.txt

Usage
CNN Performance Analysis

# Single model, single activation - quick test
python train_and_eval.py --models resnet50 --activations sslu --epochs 10 --num_trials 1

# S2LU vs LoCLU comparison
python train_and_eval.py --models resnet50 --activations sslu cahlu --epochs 20 --num_trials 3

# Multiple models test
python train_and_eval.py --models resnet50 vgg16 densenet121 --activations relu sslu cahlu --epochs 50

# CIFAR-100 test
python train_and_eval.py --dataset cifar100 --models resnet50 --activations sslu cahlu --epochs 100


Gradient Stability Analysis

# Deep network gradient stability analysis (500 layers)
python gradient_stability_analysis.py

# For different layer numbers, edit layers_to_test parameter:
# layers_to_test = [50]   # 50 layers
# layers_to_test = [100]  # 100 layers  
# layers_to_test = [500]  # 500 layers

Parameters

--dataset: Dataset selection (cifar10 or cifar100)
--models: Models to test
--activations: Activation functions to test
--epochs: Number of training epochs (default: 100)
--batch_size: Batch size (default: automatic per dataset)
--learning_rate: Learning rate (default: 0.001)
--num_trials: Number of trials per experiment (default: 5)

Available Models

resnet50, senet18, googlenet, vgg16, densenet121, densenet_cifar, mobilenet_v1, mobilenet_v2

Available Activation Functions

relu, leaky_relu, elu, gelu, mish, swish, sslu, cahlu

Quick Test Examples

#  Very quick test 
python train_and_eval.py --models resnet50 --activations sslu --epochs 5 --num_trials 1 --batch_size 64

#  Comparison test  
python train_and_eval.py --models resnet50 vgg16 --activations relu sslu cahlu --epochs 10 --num_trials 1

#  Full academic test 
python train_and_eval.py --models resnet50 senet18 vgg16 densenet121 --activations relu sslu cahlu --epochs 100 --num_trials 5

References

S. R. Dubey, S. K. Singh, and B. B. Chaudhuri, "Activation functions in deep learning: A comprehensive survey and benchmark," arXiv preprint arXiv:2109.14545v3, Jun. 28, 2022. [Online]. Available: https://arxiv.org/abs/2109.14545
Original repository: https://github.com/shivram1987/ActivationFunctions 
