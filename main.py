import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from demoNet import DemoNet
import torch.nn as nn
from python_pfm import *
import cv2


def main():
    height = 256
    width = 512
    maxdis = 160
    epoch_total = 100000000
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    disL = Variable(torch.FloatTensor(1).cuda())
    loss_mul_list = []
    for d in range(maxdis):
        loss_mul_temp = Variable(torch.Tensor(np.ones([1, 1, height, width]) * d)).cuda()
        loss_mul_list.append(loss_mul_temp)
    loss_mul = torch.cat(loss_mul_list, 1)
    loss_fn = nn.L1Loss()
    net = DemoNet()
    devices = torch.cuda.current_device()
    net = net.to(devices)
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    for epoch in range(epoch_total):
        imgL = cv2.imread('./data/left.webp').reshape(540, 960, 3)
        imgR = cv2.imread('./data/right.webp').reshape(540, 960, 3)
        imgL = transform(imgL)
        imgR = transform(imgR)
        imgL = imgL.reshape((1, 3, 540, 960))
        imgR = imgR.reshape((1, 3, 540, 960))
        dispL = readPFM('./data/disp.pfm')[0].astype(np.uint8).reshape(540, 960, 1).transpose((2, 0, 1))
        dispL = dispL.reshape((1, 1, 540, 960))
        dispL = torch.from_numpy(dispL)
        # randomH = np.random.randint(0, 160)
        # randomW = np.random.randint(0, 400)
        randomH = 100
        randomW = 200
        imgL = imgL[:, :, randomH:(randomH + height), randomW:(randomW + width)]
        imgR = imgR[:, :, randomH:(randomH + height), randomW:(randomW + width)]
        dispL = dispL[:, :, randomH:(randomH + height), randomW:(randomW + width)]
        imL.resize_(imgL.size()).copy_(imgL)
        imR.resize_(imgR.size()).copy_(imgR)
        disL.resize_(dispL.size()).copy_(dispL)
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        print(imL.shape)
        x = net(imL, imR)
        # x.resize_((1, 1, height, width))
        print(x.shape)
        x = x.view(1, 1, height, width)
        # result = torch.sum(x.mul(loss_mul), 1)
        tt = loss_fn(x, disL)
        tt.backward()
        optimizer.step()
        diff = torch.abs(x.data.cpu() - dispL.data.cpu())
        accuracy = torch.sum(diff < 3) / float(height * width * 1)
        print('=======loss value for every step=======:', epoch)
        print('=======loss value for every step=======:%f' % (tt.data))
        print('====accuracy for the result less than 3 pixels===:%f' % accuracy)


if __name__ == '__main__':
    main()
