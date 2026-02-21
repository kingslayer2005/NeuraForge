# NeuroForage

## Neural Network From Scratch (NumPy Only)

NeuroForage is a neural network framework implemented entirely from first principles using pure NumPy.

No PyTorch.  
No TensorFlow.  
No autograd.

The goal of this project is to deeply understand and extend the mathematical foundations of neural networks by building every component manually.

---

## 🔍 Project Objective

Most deep learning implementations abstract away the core mathematics.

NeuroForage focuses on:

- Manual forward propagation  
- Manual backpropagation  
- Matrix-based gradient computation  
- Custom training loop implementation  
- Full architectural control  

This project is built as a foundation for developing an adaptive, self-evolving neural architecture.

---

## 🧠 Current Version (v1)

Implemented features:

- Fully Connected (Dense) layers  
- He initialization  
- ReLU activation  
- Softmax + Cross Entropy loss  
- Manual backpropagation  
- Stochastic Gradient Descent (SGD)  
- Accuracy tracking  
- Training loss visualization  

All gradients are derived and implemented manually using matrix operations.

---

## 🏗 Architecture (v1)

Input → Dense → ReLU → Dense → Softmax → Cross Entropy Loss  

Every forward and backward step is explicitly implemented.

---

## 📈 Sample Training Output
Epoch 0, Loss: 0.9211, Accuracy: 0.5800
Epoch 100, Loss: 0.9082, Accuracy: 0.5950
Epoch 200, Loss: 0.8955, Accuracy: 0.6000
...
Epoch 900, Loss: 0.8214, Accuracy: 0.6450

Training loss decreases steadily, confirming correct gradient flow and parameter updates.

---

## 🛠 Tech Stack

- Python
- NumPy
- Matplotlib

---

## 🚀 Roadmap

Planned future versions:

- Trainable activation function (ForageAct)
- Custom optimizer (NeuroGrad++)
- Dynamic neuron growth mechanism
- Structured gradient-based pruning
- Experimental benchmarking
- Ablation studies

The long-term objective is to develop NeuroForage into a self-adaptive neural architecture.

---

## 📌 Author

Aarush Gupta

