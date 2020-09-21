import torch.nn as nn
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, structure, input_shape):
        super(Generator,self).__init__()
        self.structure =structure
        self.structure.reverse()
        print(self.structure)
        self.input_shape = input_shape
        self.input_layer=nn.Sequential(
            nn.Linear(784, 5184),
            nn.ReLU(inplace=True),
            )
        def make_layers(self, structure):
            layers=[]
            conv2d=nn.Conv2d(192, 24, kernel_size=3)
            upsample=nn.Upsample(scale_factor=1.5)
            relu=nn.ReLU(inplace=True)
            layers += [conv2d]
            for i in structure:
                if i == 'M':
                    pass
                else:
                    conv2d=nn.Conv2d(i, int(i/2), kernel_size=3)
                    layers += [conv2d]
            return layers
        self.hidden_layers=nn.Sequential(*make_layers(self, self.structure))
        self.output_layer=nn.Sequential(nn.Tanh(), nn.Conv2d(self.structure[0],3, kernel_size=3))

    def forward(self,x):
        x = self.input_layer(x)
        x = x.reshape(24,-1,3,3)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, structure, input_shape):
        super(Discriminator, self).__init__()
        self.structure=structure
        self.input_shape= input_shape
        def make_layers(self, structure, input_shape):
            layers=[]
            output=structure[0]
            conv2d=nn.Conv2d(input_shape, output, kernel_size=3)
            maxpooling=nn.MaxPool2d(kernel_size=2)
            relu=nn.ReLU(inplace=True)
            layers += [conv2d]
            for i in structure:
                if i == 'M':
                    layers += [maxpooling]
                else:
                    conv2d=nn.Conv2d(i, int(i*2), kernel_size=3)
                    layers += [conv2d, relu]
            output = i
            return layers, output
        self.avgpool=nn.AdaptiveAvgPool2d(7)
        self.hidden_layers, self.output=make_layers(self, self.structure, self.input_shape)
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_layer=nn.Sequential(
            nn.Linear(7*self.output, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 10)
        )

    def forward(self,x):
        x =self.hidden_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.output_layer(x)
        return x

img_size = (28,28)
batch_size = 8
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28,28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
    )
n_epochs=10
structure=[32,'M',64,'M',128,256]
cuda = True if torch.cuda.is_available() else False
adversarial_loss = torch.nn.BCELoss()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

generator = Generator(structure, img_size[0]*img_size[1])
discriminator = Discriminator(structure, img_size[0]*img_size[1])
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
generator
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 28*28))))
        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
