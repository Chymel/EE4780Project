import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

content_img = Image.open('C:/Users/judyd/Documents/LSU/Spring 2022/tanjiro.jpg')
style_img = Image.open('C:/Users/judyd/Documents/LSU/Spring 2022/starrynight.jpg')

# content image information
plt.imshow(content_img)
plt.grid(False)
plt.show()

# style image information
plt.imshow(style_img)
plt.grid(False)
plt.show()

# Image transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((400, 400)),
                                transforms.Normalize((0.485,0.456, 0.406),(0.229,0.224, 0.225))
                                ])
img_tensor = transform(content_img)
img_tensortwo =transform(style_img)

print(img_tensor)
print(img_tensortwo)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128	#128 is a smaller size if there is no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),	#this scales the imported image
    transforms.ToTensor()])		#transformation to a torch sensor


def image_loader(image_name):
    image = Image.open(image_name)	#this is a fake batch dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader('C:/Users/judyd/Documents/LSU/Spring 2022/starrynight.jpg')	#insert image name here
content_img = image_loader('C:/Users/judyd/Documents/LSU/Spring 2022/tanjiro.jpg')


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
