import numpy as np
import math
from copy import deepcopy
from PIL import Image


def mirror(img_i, axis):
    tmp = deepcopy(img_i)
    h = img_i.shape[0]
    w = img_i.shape[1]
    i = 0
    if axis == 'y':
        while i < h:
            j = 0
            while j < w:
                tmp[i, j] = img_i[h - i - 1, j]
                j += 1
            i += 1
    elif axis == 'x':
        while i < h:
            j = 0
            while j < w:
                tmp[i, j] = img_i[i, w - j - 1]
                j += 1
            i += 1
    return tmp


def cont(img_1, pad):
    # расширение зеркалированием по каналам
    h = img_1.shape[0]
    w = img_1.shape[1]
    if len(img_1.shape) == 3:
        help_6 = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=img_1.dtype)
    else:
        help_6 = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img_1.dtype)
    help_6[0:pad, 0: pad] = np.rot90(img_1[0: pad, 0: pad], 2, (1, 0))  # left up corner
    help_6[pad:h + pad, 0:pad] = mirror(img_1[0:h, 0:pad], 'x')  # left centre
    help_6[h + pad: h + 2 * pad, 0:pad] = np.rot90(img_1[h - pad:h, 0:pad], 2, (1, 0))  # left down corner
    help_6[0:pad, pad: w + pad] = mirror(img_1[0:pad, 0:w], 'y')  # up centre
    help_6[pad:pad + h, pad: pad + w] = deepcopy(img_1[:, :])  # main
    help_6[h + pad:h + 2 * pad, pad:w + pad] = mirror(img_1[h - pad:h, 0:w], 'y')  # down centre
    help_6[0:pad, w + pad:w + 2 * pad] = np.rot90(img_1[0: pad, w - pad: w], 2, (1, 0))  # right up corner
    help_6[pad:h + pad, w + pad:(w + (2 * pad))] = mirror(img_1[0:h, w - pad:w], 'x')  # right centre
    help_6[h + pad:h + 2 * pad, w + pad:w + 2 * pad] = np.rot90(img_1[h - pad:h, w - pad:w], 2,
                                                                (1, 0))  # right down corner
    return help_6


def weight(pad, sigma_s, sigma_r, window):
    W = np.zeros(window.shape, dtype='float32')
    for i in range(0, 2 * pad):
        for j in range(0, 2 * pad):
            W[i, j] = math.exp(-(i ** 2 + j ** 2) / (2 * (sigma_s ** 2)) - ((window[i, j] - window[pad, pad]) ** 2) / (
                    2 * (sigma_r ** 2)))  # math.exp(-(i**2 + j ** 2) / 2*sigma_s**2)

    return W


def weight_a(l, k, sigma_s, sigma_r, wind1, wind2):
    s = math.exp(-((l ** 2 + k ** 2) / (2 * sigma_s ** 2)) -
                 (((wind1 - wind2) ** 2) / (2 * sigma_r ** 2)))
    return s


def bilateral(cur, sigma_s, sigma_r):
    pad = round(sigma_s)
    img_res = np.zeros(cur.shape, cur.dtype)
    img_tmp = cont(cur, pad)
    # plt.imsave('cur_O.bmp', img_tmp, cmap=plt.cm.gray)
    for k in range(0, cur.shape[2]):
        lay = img_tmp[:, :, k]
        for y in range(0, cur.shape[0]):
            for x in range(0, cur.shape[1]):
                window = lay[y:y + 2 * pad + 1, x:x + 2 * pad + 1]
                W = weight(pad, sigma_s, sigma_r, window)
                img_res[y, x][k] = np.sum(window * W) / np.sum(W)
    return img_res.astype('uint8')

