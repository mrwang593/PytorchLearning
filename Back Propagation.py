import torch

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 5.0, 13.0, 20.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True


def forward(x):
    return w1 * x ** 2 + w2 * x


def loss(x, y):
    y_hat = forward(x)
    return (y - y_hat) ** 2


print("predict (before training)", 5, forward(5).item())

for epoch in range(100):
    epoch_loss = 0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad:", x, y, "w1", w1.grad.item(), "w2", w2.grad.item())

        w1.data -= 0.0001 * w1.grad.data
        w2.data -= 0.0001 * w2.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()

        epoch_loss += l.item()

    print("epoch:", epoch, "w1", w1.item(), "w2", w2.item(), "l", epoch_loss / len(x_data))

print("predict (after training)", 5, forward(5).item())