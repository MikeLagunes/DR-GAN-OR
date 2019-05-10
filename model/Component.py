import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
import torch.utils.model_zoo as model_zoo

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
weights = model_zoo.load_url(model_urls['resnet50'])

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def Tensor2Image(img):
    """
    input (FloatTensor)
    output (PIL.Image)
    """
    img = img.cpu()
    img = img * 0.5 + 0.5
    img = transforms.ToPILImage()(img)
    return img

def one_hot(label, depth):
    """
    Return the one_hot vector of the label given the depth.
    Args:
        label (LongTensor): shape(batchsize)
        depth (int): the sum of the labels

    output: (FloatTensor): shape(batchsize x depth) the label indicates the index in the output

    >>> label = torch.LongTensor([0, 0, 1])
    >>> one_hot(label, 2)
    <BLANKLINE>
     1  0
     1  0
     0  1
    [torch.FloatTensor of size 3x2]
    <BLANKLINE>
    """
    out_tensor = torch.zeros(len(label), depth)
    for i, index in enumerate(label):
        out_tensor[i][index] = 1
    return out_tensor

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class conv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 3, 96, 96))

    >>> net = conv_unit(3, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 96, 96])

    >>> net = conv_unit(3, 16, pooling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 48, 48])
    """

    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Fconv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 64, 48, 48))

    >>> net = Fconv_unit(64, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 48, 48])

    >>> net = Fconv_unit(64, 16, unsampling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 96, 96])
    """

    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()

        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Decoder(nn.Module):
    """
    Args:
        N_z (int): The dimensions of the noise

    >>> Dec = Decoder()
    >>> input = Variable(torch.randn(4, 372))
    >>> output = Dec(input)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_z=50):
        super(Decoder, self).__init__()
        Fconv_layers = [
            Fconv_unit(320, 160),                   #Bx160x6x6
            Fconv_unit(160, 256),                   #Bx256x6x6
            Fconv_unit(256, 256, unsampling=True),  #Bx256x12x12
            Fconv_unit(256, 128),                   #Bx128x12x12
            Fconv_unit(128, 192),                   #Bx192x12x12
            Fconv_unit(192, 192, unsampling=True),  #Bx192x24x24
            Fconv_unit(192, 96),                    #Bx96x24x24
            Fconv_unit(96, 128),                    #Bx128x24x24
            Fconv_unit(128, 128, unsampling=True),  #Bx128x48x48
            Fconv_unit(128, 64),                    #Bx64x48x48
            Fconv_unit(64, 64),                     #Bx64x48x48
            Fconv_unit(64, 64, unsampling=True),    #Bx64x96x96
            Fconv_unit(64, 32),                     #Bx32x96x96
            Fconv_unit(32, 3)                       #Bx3x96x96
        ]

        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        self.fc = nn.Linear(320+N_z, 320*6*6)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 320, 6, 6)
        x = self.Fconv_layers(x)
        return x

class Multi_Encoder(nn.Module):
    """
    The multi version of the Encoder.

    >>> Enc = Multi_Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    def __init__(self):
        super(Multi_Encoder, self).__init__()
        conv_layers = [
            conv_unit(3, 32),                   #Bx32x96x96
            conv_unit(32, 64),                  #Bx64x96x96
            conv_unit(64, 64, pooling=True),    #Bx64x48x48
            conv_unit(64, 64),                  #Bx64x48x48
            conv_unit(64, 128),                 #Bx128x48x48
            conv_unit(128, 128, pooling=True),  #Bx128x24x24
            conv_unit(128, 96),                 #Bx96x24x24
            conv_unit(96, 192),                 #Bx192x24x24
            conv_unit(192, 192, pooling=True),  #Bx192x12x12
            conv_unit(192, 128),                #Bx128x12x12
            conv_unit(128, 256),                #Bx256x12x12
            conv_unit(256, 256, pooling=True),  #Bx256x6x6
            conv_unit(256, 160),                #Bx160x6x6
            conv_unit(160, 321),                #Bx321x6x6
            nn.AvgPool2d(kernel_size=6)         #Bx321x1x1
        ]

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 321)
        t = x[:, :320]
        w = x[:, 320]
        batchsize = len(w)
        r = Variable(torch.zeros(t.size())).type_as(t)
        for i in range(batchsize):
            r[i] = t[i] * w[i]
        r = torch.sum(r, 0, keepdim=True).div(torch.sum(w))

        return torch.cat((t,r.type_as(t)), 0)


class Encoder(nn.Module):
    """
    Encoder with ResNet-50.

    >>> Enc = Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_bottleneck = self.fc(x)

        return x_bottleneck # Returns an embedding of 320 elements



class Encoder_old(nn.Module):
    """
    The single version of the Encoder.

    >>> Enc = Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    def __init__(self):
        super(Encoder, self).__init__()
        conv_layers = [
            conv_unit(3, 32),                   #Bx32x96x96
            conv_unit(32, 64),                  #Bx64x96x96
            conv_unit(64, 64, pooling=True),    #Bx64x48x48
            conv_unit(64, 64),                  #Bx64x48x48
            conv_unit(64, 128),                 #Bx128x48x48
            conv_unit(128, 128, pooling=True),  #Bx128x24x24
            conv_unit(128, 96),                 #Bx96x24x24
            conv_unit(96, 192),                 #Bx192x24x24
            conv_unit(192, 192, pooling=True),  #Bx192x12x12
            conv_unit(192, 128),                #Bx128x12x12
            conv_unit(128, 256),                #Bx256x12x12
            conv_unit(256, 256, pooling=True),  #Bx256x6x6
            conv_unit(256, 160),                #Bx160x6x6
            conv_unit(160, 320),                #Bx320x6x6
            nn.AvgPool2d(kernel_size=6)         #Bx320x1x1
        ]

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 320)
        return 


class Generator(nn.Module):
    """
    >>> G = Generator()

    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> pose = Variable(torch.randn(4, 2))
    >>> noise = Variable(torch.randn(4, 50))

    >>> output = G(input, pose, noise)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_z=50, single=False):
        super(Generator, self).__init__()
        if single:
            self.enc = Encoder(Bottleneck, [3, 4, 6, 3], num_classes=1000)
            self.enc.load_state_dict(weights)
            self.enc = nn.Linear(96, 320)
            #self.enc = nn.Linear(2048, 320)
        else:
            self.enc = Multi_Encoder()

        self.dec = Decoder(N_z)


    def forward(self, input, noise):
        x = self.enc(input)
        #print('{0}/t{1}/t{2}'.format(x.size(), pose.size(), noise.size()))
        x = torch.cat((x, noise), 1)
        x = self.dec(x)
        return x

class Discriminator(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_d (int): The sum of the identities

    >>> D = Discriminator()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = D(input)
    >>> output.size()
    torch.Size([4, 503])
    """
    def __init__(self, n_classes=10):
        super(Discriminator, self).__init__()
        conv_layers = [
            conv_unit(3, 32),                   #Bx32x96x96
            conv_unit(32, 64),                  #Bx64x96x96
            conv_unit(64, 64, pooling=True),    #Bx64x48x48
            conv_unit(64, 64),                  #Bx64x48x48
            conv_unit(64, 128),                 #Bx128x48x48
            conv_unit(128, 128, pooling=True),  #Bx128x24x24
            conv_unit(128, 96),                 #Bx96x24x24
            conv_unit(96, 192),                 #Bx192x24x24
            conv_unit(192, 192, pooling=True),  #Bx192x12x12
            conv_unit(192, 128),                #Bx128x12x12
            conv_unit(128, 256),                #Bx256x12x12
            conv_unit(256, 256, pooling=True),  #Bx256x6x6
            conv_unit(256, 160),                #Bx160x6x6
            conv_unit(160, 320),                #Bx320x6x6
            nn.AvgPool2d(kernel_size=6)         #Bx320x1x1
        ]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_embedding = nn.Linear(320, 128)
        self.fc = nn.Linear(128, n_classes)

    def forward(self,input):
        x = self.conv_layers(input)
        x = x.view(-1, 320)
        x = self.fc_embedding(x)
        x = self.fc(x)
        return x
