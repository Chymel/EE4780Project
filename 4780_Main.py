import numpy as np
import torch
import os
import random
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from vgg import VGG

content_layers = ['re42']
style_layers = ['re11', 're21', 're31', 're41', 're51']

torch.cuda.manual_seed_all(random.randint(1, 1000))
if not os.path.exists("images/"):
    os.makedirs("images/")

cudnn.benchmark = True



# Image transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((400, 400)),
                                transforms.Normalize((0.485,0.456, 0.406),(0.229,0.224, 0.225))
                                ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128	#128 is a smaller size if there is no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),	#this scales the imported image
    transforms.ToTensor()])		#transformation to a torch sensor


def image_loader(image_name):
    image = Image.open(image_name)	#this is a fake batch dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader('noodles.jpg')	#insert image name here
content_img = image_loader('groudon.jpg')

unloader = transforms.ToPILImage()

plt.ion()


class Picture:
    pass


def imshow(tensor, title = Picture):
    image = tensor.cpu().clone()	#we clone this to avoid changes to it
    image = image.squeeze(0)		#removes to fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not Picture:
        plt.title(title)
    plt.pause(0.005)	#delay to update plots


plt.figure()
imshow(style_img, title='Art Image')

plt.figure()
imshow(content_img, title='Original Image')


style_img = style_img.cuda()
content_img = content_img.cuda()

vgg_directory = "./vgg_conv.pth"
vgg = VGG()

vgg.load_state_dict(torch.load(vgg_directory))
for parameter in vgg.parameters():
    parameter.requires_grad = False

vgg.cuda()

############################################# Loss Calculations #############################################

class Gram_Matrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        e = input.view(a, b, c * d)
        G = torch.bmm(e, e.transpose(1, 2))
        return G.div_(c * d)

class style_Loss(nn.Module):
    def forward(self, input, target):
        GramInput = Gram_Matrix()(input)
        return nn.MSELoss()(GramInput, target)

styleTarget = []
for i in vgg(style_img, style_layers):
    i = i.detach()
    styleTarget.append(Gram_Matrix()(i))

contentTarget = []
for i in vgg(content_img, content_layers):
    i = i.detach()
    contentTarget.append(i)

styleLosses = [style_Loss()] * len(style_layers)

contentLosses = [nn.MSELoss()] * len(content_layers)

losses = styleLosses + contentLosses
targets = styleTarget + contentTarget
loss_layers = style_layers + content_layers
style_weight = 0.5
content_weight = 0.5

weights = [style_weight] * len(style_layers) + [content_weight] * len(content_layers)

for every_loss in losses:
    every_loss = every_loss.cuda()

#############################################################################################################
