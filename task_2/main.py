import sys
from PIL import Image
import numpy as np

from Bilateral_filter import bilateral
from first_part import discrete_gradient, non_max2, hysteresis, threshold, non_max_suppression


def main():
    lensys = len(sys.argv)
    command = sys.argv[1]
    sigma = float(sys.argv[2])
    img_in = sys.argv[lensys - 2]
    img_out = sys.argv[lensys - 1]
    image = Image.open(img_in)
    image_L = image.convert('L')
    img = np.asarray(image_L, dtype='float32')
    if command == "dir":
        res = discrete_gradient(img, sigma)[0]
    elif command == "nonmax":
        res = non_max2(img, sigma)
    elif command == "canny":
        img_1 = threshold(non_max2(img, sigma),  float(sys.argv[4]), float(sys.argv[3]))
        res = hysteresis(img_1)
    elif command == "bilateral":
        img_B = np.asarray(image, dtype='float32')
        res = bilateral(img_B, float(sys.argv[2]), float(sys.argv[3]))
    img_res = Image.fromarray(res)
    img_res.save(img_out)
    return 0


if __name__ == "__main__":
    main()
