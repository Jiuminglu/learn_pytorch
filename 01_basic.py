import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms

caption = """
# ========================================================
#  1. Basic autograd example 1
# ========================================================
"""
print(caption)

x = torch.tensor(1.0, requires_grad = True)
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(3.0, requires_grad = True)

y = w * x + b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

caption = """
# ========================================================
#  2. Basic autograd example - linear regression
# ========================================================
"""
print(caption)

x_input  = torch.randn(10, 3)
y_actual = torch.randn(10, 2)

linear = nn.Linear(3, 2)
w = linear.weight
b = linear.bias

print(w)
print(b)

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
pred = linear(x_input)
loss = loss_function(pred, y_actual)

print('loss: ', loss.item())
loss.backward()

print('dL/dw = ', linear.weight.grad)
print('dL/db = ', linear.bias.grad)

optimizer.step()
pred = linear(x_input)
loss = loss_function(pred, y_actual)
print('loss after 1 step optimization: ')
print("loss = ", loss.item())



caption = """
# ========================================================
#  3. Loading data from numpy
# ========================================================
"""
print(caption)

x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()

print(x)
print(y)
print(z)

