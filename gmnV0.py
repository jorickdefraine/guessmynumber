import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from data.examples.samples import *
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

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
def process_image(img):
    # Load Image
    if type(img) == str:
        img = Image.open(image_path)

    img = img.convert('1')
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
    img = (img-0.1307)/0.3081
    # Normalize based on the preset mean and standard deviation
    #img[0] = (img[0] - 0.1307) / 0.3081
    #img[1] = (img[1] - 0.1307) / 0.3081
    #img[2] = (img[2] - 0.1307) / 0.3081
    # Add a fourth dimension to the beginning to indicate batch size
    # img = img[np.newaxis, :]
    #new_img = img[np.newaxis, np.newaxis, :]
    # Turn into a torch tensor
    image = Image.fromarray(img, '1')
    #image = torch.from_numpy(img)
    ##image = image.unsqueeze(0).unsqueeze(0)
    #image = image.float()

    return image


def predict(image):
    network = Net()

    network.load_state_dict(torch.load("./results/model.pth"))

    #network = torch.load("./results/model.pth")
    network.eval()
    #network = network.eval()

    # Pass the image through our model
    output = network(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()


def stringtotensor(input):
    position = input[14:-2].replace('{', '').replace('}', '').replace('"', '') \
        .replace(':', '').replace(',', '').replace('x', ' ').replace('y', ' ').split(' ')
    position = [int(i) for i in position]
    position = np.array(position).reshape((int(len(position) / 2), 2))

    w, h = 280, 280
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for element in position:
        for i in range(30):
            output[element[0]:element[0]+i,element[1]:element[1]+i] = [255, 255, 255]



    img = Image.fromarray(output, 'RGB').convert('1')
    img_mirror = ImageOps.mirror(img)
    transposed = img_mirror.transpose(Image.ROTATE_90)
    return transposed
def predictwithkeras(image_path):

    # Test on the image form test folders
    img_width, img_height = 28, 28
    test_model = load_model("./results/conv2D_classifier.h5")
    img = load_img(image_path, False, target_size=(img_width, img_height), color_mode="grayscale")
    print(type(img))
    # img.show()
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = test_model.predict_classes(x)
    prob = test_model.predict_proba(x)
    prob2 = test_model.predict(x)
    return prob, preds

if __name__ == '__main__':
    # main()
    #image_path = "./data/examples/image.png"
    ## input = '{"points":[{"x":33,"y":29},{"x":34,"y":29},{"x":34,"y":28},{"x":35,"y":28},{"x":36,"y":28}]}'
    #image = process_image(image_path)
    #print(image.shape)
    ## print(stringtotensor(input))
    #output = predict(image)
    #print(output)

    drawingJSON = '{"points":[{"x":100,"y":91},{"x":98,"y":90},{"x":96,"y":90},{"x":95,"y":90},{"x":91,"y":90},{"x":89,"y":90},{"x":85,"y":88},{"x":82,"y":87},{"x":80,"y":86},{"x":77,"y":84},{"x":76,"y":83},{"x":75,"y":81},{"x":74,"y":79},{"x":74,"y":77},{"x":74,"y":73},{"x":75,"y":71},{"x":76,"y":70},{"x":77,"y":68},{"x":79,"y":67},{"x":81,"y":65},{"x":85,"y":63},{"x":89,"y":61},{"x":93,"y":60},{"x":96,"y":60},{"x":100,"y":59},{"x":102,"y":58},{"x":106,"y":58},{"x":109,"y":58},{"x":113,"y":58},{"x":121,"y":60},{"x":124,"y":62},{"x":126,"y":64},{"x":127,"y":66},{"x":128,"y":70},{"x":129,"y":74},{"x":130,"y":80},{"x":130,"y":85},{"x":129,"y":91},{"x":128,"y":96},{"x":125,"y":102},{"x":123,"y":113},{"x":120,"y":119},{"x":118,"y":124},{"x":116,"y":128},{"x":114,"y":131},{"x":112,"y":134},{"x":110,"y":137},{"x":107,"y":141},{"x":104,"y":146},{"x":102,"y":148},{"x":100,"y":149},{"x":99,"y":150},{"x":98,"y":150},{"x":96,"y":150},{"x":95,"y":150},{"x":92,"y":150},{"x":91,"y":150},{"x":90,"y":150},{"x":87,"y":149},{"x":84,"y":147},{"x":80,"y":145},{"x":79,"y":144},{"x":76,"y":144},{"x":74,"y":143},{"x":72,"y":141},{"x":72,"y":140},{"x":71,"y":139},{"x":71,"y":138},{"x":71,"y":136},{"x":73,"y":135},{"x":74,"y":134},{"x":77,"y":134},{"x":80,"y":133},{"x":84,"y":133},{"x":88,"y":133},{"x":93,"y":135},{"x":96,"y":136},{"x":99,"y":138},{"x":101,"y":139},{"x":103,"y":140},{"x":105,"y":141},{"x":106,"y":142},{"x":108,"y":144},{"x":109,"y":144},{"x":111,"y":145},{"x":113,"y":147},{"x":118,"y":149},{"x":121,"y":150},{"x":126,"y":151},{"x":130,"y":152},{"x":135,"y":152},{"x":136,"y":152},{"x":140,"y":152},{"x":147,"y":152},{"x":152,"y":152},{"x":158,"y":152},{"x":163,"y":152},{"x":167,"y":152},{"x":169,"y":152},{"x":170,"y":152},{"x":171,"y":152},{"x":171,"y":151}]}'
    pre_image = stringtotensor(neuf)
    pre_image.show()
    #image = process_image(pre_image)
    #image.show()
    pre_image.save('./data/examples/pre_image.png')
    # print(pre_image)
    ## print(image)
    #output = predict(image)
    #print(output)
    path = './data/examples/pre_image.png'
    print(predictwithkeras(path))