# Lab 2 · Automatic Differentiation & Backpropagation

How PyTorch computes gradients with `autograd`, and how backpropagation is used to train a small neural network.

**File:** [`02_autograd_backpropagation.py`](02_autograd_backpropagation.py). It is a Colab notebook exported as a Python script, so the Markdown cells appear as comments and docstrings.

## What I worked on

- Tracking gradients with `requires_grad` and calling `.backward()` on a computational graph
- Backpropagation with explicit gradients and `retain_graph`
- Defining a 2-layer network (`nn.Linear(3, 20)` → `nn.Linear(20, 3)`) as an `nn.Module`
- Writing the training loop with `nn.MSELoss` and SGD, including where `optimizer.zero_grad()` belongs
- Comparing trained and untrained outputs, and studying the effect of `detach()` on gradient flow

## Stack

Python · PyTorch · Matplotlib · NumPy · Google Colab

## How to run

Open the script in [Google Colab](https://colab.research.google.com/) or run it locally:

```bash
pip install torch numpy matplotlib
python 02_autograd_backpropagation.py
```

## Credits

Lab template created by Santiago Pascual (UPC School, 2019), updated by Xavier Giró-i-Nieto as course material. The original authors are credited at the top of the file. The exercise solutions are my own work.

---
Part of my [GDSA labs](https://github.com/Pedrom-GH/GDSA_PEDRO) · B.Sc. in Telecommunications Systems Engineering, UPC – ESEIAAT · Pedro Muñoz Álvarez
