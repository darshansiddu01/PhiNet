import torch
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps
resize_shape = [128, 256]
from torchvision import transforms
from matplotlib import pyplot as plt

ToGray = transforms.Grayscale()
T = transforms.ToTensor()


def PreProcess(image):
    '''
    nparr = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    print(img)
    img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
    print(img)
    img.astype(np.float64)
    print(img)
    img = np.transpose(img[np.newaxis, np.newaxis, :, :], (0, 1, 3, 2))
    print(img)
    print((torch.from_numpy(img.copy()).float())/255.00)
    return (torch.from_numpy(img.copy()).float())/255.00


'''
    image.save(image.filename)
    img = ToGray(Image.open(image.filename))
    #img = ImageOps.invert(img)
    #plt.imshow(T(img)[0, :, :])
    # img.show()
    # plt.waitforbuttonpress()
    # plt.show()
    # print(T(img))
    return T(img.resize(resize_shape))[np.newaxis, :, :, :]


def EC_dist(x1, x2):
    #print(x1.shape, x2.shape)
    return torch.sum(torch.pow(F.pairwise_distance(x1, x2), 2))


class ConvNet(nn.Module):
    def __init__(self, ):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.3))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1))

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.3))

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.3))

        self.adap = nn.AdaptiveAvgPool3d((128, 6, 6))

        self.layer6 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer7 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU())

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        # print (out.size())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.adap(out)
        # print (out.size())
        out = out.reshape(out.size()[0], -1)

        out = self.layer6(out)

        out = self.layer7(out)
        out = self.sigmoid(out)
        return out
