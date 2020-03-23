import numpy as np
import matplotlib as plt
from PIL import Image
#from copy import deepcopy
import scipy as scp

def dif_gauss_kernel(sigma):
    size = int(round(sigma) * 3) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    tmp_x, tmp_y =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    for i in range(-size,size + 1):
        for j in range(-size,size + 1):
            tmp_y[i+size,j+size] *= -j/(sigma*sigma)
            tmp_x[i+size,j+size] *= -i/(sigma*sigma)
    return tmp_y, tmp_x

def discrete_gradient(img, sigma): #работа в градациях серого
    img_tmp = img.astype('float32')
    dif_y_Gaus, dif_x_Gaus = dif_gauss_kernel(sigma)
    img_res = np.zeros(img_tmp.shape, dtype='uint8')
    img_y = scp.convolve(img_tmp, dif_y_Gaus, mode='reflect')
    img_x = scp.convolve(img_tmp, dif_x_Gaus, mode='reflect')
    theta = np.arctan2(img_y, img_x)
    theta = theta * 180. / np.pi
    theta[theta < 0] += 180
    for i in range(img.shape[0]):
        for j in range (img.shape[1]):
            if (img_x[i,j] == img_y[i,j] == 0):
                img_res[i,j] = 0
            elif (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                img_res[i,j] = 64
            elif (22.5 <= theta[i,j] < 67.5):
                img_res[i, j] = 192
            elif (67.5 <= theta[i,j] < 112.5):
                img_res[i, j] = 128
            elif (112.5 <= theta[i,j] < 157.5):
                img_res[i, j] = 255
    return img_res, theta

def non_max_suppression(img, sigma):
    img_tmp, theta = discrete_gradient(img, sigma)
    img_res = np.zeros(img.shape, dtype = 'uint8')

    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            q,r = 255           #присваиваем max значения
                                #вычисляем направление grad
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                q = img_tmp[i, j + 1]
                r = img_tmp[i, j - 1]
            elif (22.5 <= theta[i, j] < 67.5):
                q = img_tmp[i - 1, j + 1]
                r = img_tmp[i + 1, j - 1]
            elif (67.5 <= theta[i, j] < 112.5):
                q = img_tmp[i - 1, j]
                r = img_tmp[i + 1, j]
            elif (112.5 <= theta[i, j] < 157.5):
                q = img_tmp[i - 1, j - 1]
                r = img_tmp[i + 1, j + 1]

            if (img_tmp[i, j] >= q) and (img_tmp[i, j] >= r):
                img_res[i, j] = img_tmp[i, j]
            else:
                img_res[i, j] = 0

    return img_res

image = Image.open('lena.bmp')
image_2 = image.convert('L')
img = np.asarray(image_2, dtype='uint8')
h = image.size[0]
w = image.size[1]
img = img.astype('float32')
#myrot = median(image, 2)
#plt.imshow(myrot)
img_2 = discrete_gradient(img, 1)
#plt.imshow(img_2, cmap = 'gray')
#plt.imshow(img_2, cmap = 'gray')
#myGx = SobelX(img)
#myGy = SobelY(image)
#plt.imshow(myGx, cmap = 'gray')
#plt.imshow(myGy)
#print(myGx.shape)
#image_X = Image.fromarray(myrot)
image = Image.fromarray(img_2)
#image_X.save("SX.bmp")
image.save("output.bmp")








