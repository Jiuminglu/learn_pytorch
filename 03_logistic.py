import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


train_dataset = torchvision.datasets.MNIST(root='data/', train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
test_dataset  = torchvision.datasets.MNIST(root='data/', train=False,
                                            transform=transforms.ToTensor())


input_size  = 784
num_classes = 10
batch_size  = 100
num_epochs  = 5
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

# Model
model = nn.Linear(input_size,  num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)

        output = model(images)
        loss   = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch: {}, batch: {}, loss: {:.4f}".format(epoch, i, loss.item()))
