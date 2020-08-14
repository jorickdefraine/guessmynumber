import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(network, train_loader, optimizer, epoch, train_losses, train_counter, log_interval):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
            train_losses.append(loss)
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    n_epochs = 14
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                               (0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(network, train_loader, optimizer, epoch, train_losses, train_counter, log_interval)
        test(network, test_loader, test_losses)


# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path).convert('1')

    # Get the dimensions of the image
    width, height = img.size
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((28, int(28 * (height / width))) if width < height else (int(28 * (width / height)), 28))

    # Get the dimensions of the new image size
    width, height = img.size

    # Set the coordinates to do a center crop of 28 x 28
    left = (width - 28) / 2
    top = (height - 28) / 2
    right = (width + 28) / 2
    bottom = (height + 28) / 2
    img = img.crop((left, top, right, bottom))

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((0, 1))

    # Make all values between 0 and 1
    img = img / 28

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.1307) / 0.3081
    img[1] = (img[1] - 0.1307) / 0.3081
    #img[2] = (img[2] - 0.1307) / 0.3081

    # Add a fourth dimension to the beginning to indicate batch size
    #img = img[np.newaxis, :]
    img = img[np.newaxis, np.newaxis, :]
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


def predict(image, network):
    network = network.eval()

    # Pass the image through our model
    output = network(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

if __name__ == '__main__':
    #main()
    image_path = "./data/examples/image.png"
    image = process_image(image_path)
    network = Net()
    print(image.shape)
    network.load_state_dict(torch.load('./results/model.pth'))
    top_prob, top_class = predict(image, network)
    print(top_prob, top_class)
    #model = Net()
    #model.load_state_dict(torch.load('./results/model.pth'))

    #output = model(input)
    #prediction = int(torch.max(output.data, 1)[1].numpy())
    #print(prediction)
