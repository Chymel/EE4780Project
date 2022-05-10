import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision.utils as tutils
from vgg import VGG
import torch.optim as optim
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
import cv2
from tqdm import tqdm

content_layers = ['re42']
style_layers = ['re11', 're21', 're31', 're41', 're51']

torch.cuda.manual_seed_all(random.randint(1, 1000))
if not os.path.exists("images/outputImages"):
    os.makedirs("images/outputImages")

cudnn.benchmark = True


# Image transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 400 if torch.cuda.is_available() else 128	#128 is a smaller size if there is no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),	#this scales the imported image
    transforms.ToTensor()]) #transformation to a torch sensor

def image_loader(image_name):
    image = Image.open(image_name)	#this is a fake batch dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

styleFileName0 = input("Enter the name of the image you want to use for your style image >> ")
styleFileName = "images/styleImages/" + styleFileName0
contentFileName0 = input("Enter the name of the image you want to use for your content image >> ")
contentFileName = "images/contentImages/" + contentFileName0

outName = styleFileName0.replace('.jpg', '') + contentFileName0.replace('.jpg', '') + ".png"
outDir = "images/outputImages/" + outName

style_img = image_loader(styleFileName)	#insert image name here
content_img = image_loader(contentFileName)

unloader = transforms.ToPILImage()

plt.ion()

sWeight = input("Enter the Weight for the Style Image >> ")
cWeight = input("Enter the Weight for the Content Image >> ")
numIter = input("Enter the number of iterations to run >> ")

def save_img(img):
    img = img.clamp_(0, 1)
    tutils.save_image(img,
                      outDir,
                      normalize=True)

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

################## Loss Calculations ##################

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

style_weight = int(sWeight)
content_weight = int(cWeight)

numberOfIterations = int(numIter)

weights = [style_weight] * len(style_layers) + [content_weight] * len(content_layers)

#############################################################################################################

optimizeImg = Variable(content_img.data.clone(), requires_grad=True)
optimizer = optim.LBFGS([optimizeImg])

for every_loss in losses:
    every_loss = every_loss.cuda()
optimizeImg.cuda()

for i in tqdm(range(1, numberOfIterations+1), desc="Loading..."):
    def calc():
        optimizer.zero_grad()
        out = vgg(optimizeImg, loss_layers)
        totalLossList = []
        for j in range(len(out)):
            layerOutput = out[j]
            loss_j = losses[j]
            target_j = targets[j]
            totalLossList.append(loss_j(layerOutput, target_j) * weights[j])
        totalLoss = sum(totalLossList)
        totalLoss.backward()
        return totalLoss
    optimizer.step(calc)
outImg = optimizeImg.data[0].cpu()
save_img(outImg.squeeze())

plt.figure()
imshow(outImg, title=outName)

imageA = cv2.imread(styleFileName)
imageB = cv2.imread(contentFileName)
imageC = cv2.imread(outDir)

imageA = cv2.resize(imageA, (400, 400))
imageB = cv2.resize(imageB, (400, 400))
imageC = cv2.resize(imageC, (400, 400))

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
grayC = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)

scoreStyle = compare_ssim(grayA, grayC) * 100
scoreContent = compare_ssim(grayB, grayC) * 100

scoreStyle = "{:.2f}".format(scoreStyle)
scoreContent = "{:.2f}".format(scoreContent)


print('Similarity between', styleFileName0, 'and', outName, ': ', scoreStyle, '%')
print('Similarity between', contentFileName0, 'and', outName, ': ', scoreContent, '%')


