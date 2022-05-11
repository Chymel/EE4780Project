
![Logo](https://i.ibb.co/GTb1PGq/lsulogo.png)


# Intricate Art Generation Using Neural Style Transfer

The application of the “style” of a painting to a given picture using Neural Style Transfer 

![Example](https://i.ibb.co/yq2WQ5q/Screenshot-2022-03-20-202438-PNG.jpg)


## Installation

Install dependencies with pip

```bash
  pip install -r requirements.txt
```
    
## Usage


Note: This program uses CUDA and needs a CUDA capable \
GPU to use it and a CUDA enabled version of pyTorch

Make sure you put your style and content images into their \
respective directories, images/styleImages & images/contentImages

Run:
```python
python 4780_Main.py
```
The program will ask you which images you want to use:
```bash
Enter the name of the image you want to use for your style image >> picasso.jpg
Enter the name of the image you want to use for your content image >> lion.jpg
```
It will then ask you what weights you want to use:
```bash
Enter the Weight for the Style Image >> 1000
Enter the Weight for the Content Image >> 5
```
Then it asks how many iterations you want to run:
```bash
Enter the number of iterations to run >> 200
```
The program will then continue to run for however many iterations and produce\
an output image
```bash
Loading...: 100%|██████████| 200/200 [03:09<00:00,  1.05it/s]
Similarity between picasso.jpg and picassolion.png :  26.31 %
Similarity between lion.jpg and picassolion.png :  23.98 %
```

While the program runs, it shows you the different images your using and then your\
final output image:

![StyleImage](https://i.ibb.co/sCC6cP9/art.png)
![ContentImage](https://i.ibb.co/st6nZ9Y/original.png)
![OutputImage](https://i.ibb.co/HDs1tpX/final.png)
## Authors

- [Carson Hymel | https://github.com/Chymel | chyme21@lsu.edu](https://github.com/Chymel)
- [Rachel Yiu | https://github.com/ryiuready | ryiu1@lsu.edu](https://github.com/ryiuready)
- [Jacob Chandler | https://github.com/JTC-76 | jchan24@lsu.edu](https://github.com/JTC-76)
- [Elizabeth Dao | https://github.com/edao3 | edao3@lsu.edu](https://www.github.com/edao3)

## References

 - [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
