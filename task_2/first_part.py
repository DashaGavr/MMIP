import numpy as np
from PIL import Image
from scipy import ndimage
from copy import deepcopy
import matplotlib.pyplot as plt

def contrasting(lay):
    #   y = a * x + b
    if lay.max() != 0:
        lay = (255 / (lay.max())) * lay
    return lay


def dif_gauss_kernel(sigma):
    size = int(round(sigma) * 3) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    tmp_x = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    tmp_y = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal

    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            tmp_y[i + size, j + size] *= -i / (sigma * sigma)
            tmp_x[i + size, j + size] *= -j / (sigma * sigma)

    return tmp_y, tmp_x


def gradient(img_base, sigma):
    img_tmp = img_base.astype('float32')
    dif_y_Gauss, dif_x_Gauss = dif_gauss_kernel(sigma)

    img_y = ndimage.filters.convolve(img_tmp, dif_y_Gauss, mode='reflect')
    img_x = ndimage.filters.convolve(img_tmp, dif_x_Gauss, mode='reflect')

    img_res = np.hypot(img_x, img_y)

    img_res = contrasting(img_res.astype('uint8'))

    plt.imsave('cur0.bmp', img_res, cmap=plt.cm.gray)

    return img_res


def discrete_gradient(img_base, sigma):  # работа в градациях серого
    img_tmp = img_base.astype('float32')
    dif_y_Gauss, dif_x_Gauss = dif_gauss_kernel(sigma)
    img_res = np.zeros(img_tmp.shape, dtype='uint8')

    img_y = ndimage.filters.convolve(img_tmp, dif_y_Gauss, mode='reflect')
    img_x = ndimage.filters.convolve(img_tmp, dif_x_Gauss, mode='reflect')

    theta = np.arctan2(img_y, img_x)
    theta *= (180.0 / np.pi)
    theta[theta < 0] += 180

    for i in range(img_base.shape[0]):
        for j in range(img_base.shape[1]):
            if img_x[i, j] == img_y[i, j] == 0:
                img_res[i, j] = 0
            elif (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                img_res[i, j] = 64
            elif 22.5 <= theta[i, j] < 67.5:
                img_res[i, j] = 255
            elif 67.5 <= theta[i, j] < 112.5:
                img_res[i, j] = 128
            elif 112.5 <= theta[i, j] < 157.5:
                img_res[i, j] = 192

    return img_res, theta


def non_max2(img_base, sigma):
    img_tmp_1 = gradient(img_base, sigma)
    img_tmp_2 = discrete_gradient(img_base, sigma)[0]
    img_res = np.zeros(img_base.shape, dtype='uint8')

    for i in range(1, img_base.shape[0] - 1):
        for j in range(1, img_base.shape[1] - 1):
            q = 255
            r = 255
            # присваиваем max значения
            # вычисляем направление grad
            if img_tmp_2[i, j] == 192:
                q = img_tmp_1[i + 1, j - 1]
                r = img_tmp_1[i - 1, j + 1]
            elif img_tmp_2[i, j] == 64:
                q = img_tmp_1[i, j + 1]
                r = img_tmp_1[i, j - 1]
            elif img_tmp_2[i, j] == 128:
                q = img_tmp_1[i + 1, j]
                r = img_tmp_1[i - 1, j]
            elif img_tmp_2[i, j] == 255:
                q = img_tmp_1[i - 1, j - 1]
                r = img_tmp_1[i + 1, j + 1]

            if (img_tmp_1[i, j] >= q) and (img_tmp_1[i, j] >= r):
                img_res[i, j] = img_tmp_1[i, j]
            else:
                img_res[i, j] = 0

    return img_res


def non_max_suppression(img_base, sigma):
    img_tmp, theta = gradient(img_base, sigma)
    img_res = np.zeros(img_base.shape, dtype='float32')

    for i in range(1, img_base.shape[0] - 1):
        for j in range(1, img_base.shape[1] - 1):
            q = 255
            r = 255
            # присваиваем max значения
            # вычисляем направление grad
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                q = img_tmp[i, j + 1]
                r = img_tmp[i, j - 1]
            elif 22.5 <= theta[i, j] < 67.5:
                q = img_tmp[i - 1, j - 1]
                r = img_tmp[i + 1, j + 1]
            elif 67.5 <= theta[i, j] < 112.5:
                q = img_tmp[i + 1, j]
                r = img_tmp[i - 1, j]
            elif 112.5 <= theta[i, j] < 157.5:
                q = img_tmp[i + 1, j - 1]
                r = img_tmp[i - 1, j + 1]

            if (img_tmp[i, j] >= q) and (img_tmp[i, j] >= r):
                img_res[i, j] = img_tmp[i, j]
            else:
                img_res[i, j] = 0

    img_res = contrasting(img_res).astype('uint8')
    # np.clip(img_res, 0, 255, img_res)

    return img_res


def threshold(img_base, low, high):
    res = np.zeros(img_base.shape, dtype=img_base.dtype)

    high = img_base.max() * high
    low = high * low

    weak = 0
    strong = 255

    strong_i, strong_j = np.where(img_base >= high)
    weak_i, weak_j = np.where((img_base <= high) & (img_base >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    plt.imsave('cur.bmp', res, cmap=plt.cm.gray)

    return res


def hysteresis(img_base):
    weak = 0
    strong = 255
    res = np.zeros(img_base.shape, dtype='uint8')

    for i in range(1, img_base.shape[0] - 1):
        for j in range(1, img_base.shape[1] - 1):
            if img_base[i, j] == weak:
                if ((img_base[i + 1, j - 1] == strong) or (img_base[i + 1, j] == strong) or (
                        img_base[i + 1, j + 1] == strong)
                        or (img_base[i, j - 1] == strong) or (img_base[i, j + 1] == strong)
                        or (img_base[i - 1, j - 1] == strong) or (img_base[i - 1, j] == strong) or (
                                img_base[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = weak
    return res
