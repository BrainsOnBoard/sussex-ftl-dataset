import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import math


def load_images(dir, greyscale=False):

    if greyscale:
        channels = cv.IMREAD_GRAYSCALE
    else:
        channels = cv.IMREAD_COLOR

    images = []
    #dirListing = os.listdir(dir)
    # count .png files
    png_files_num = len([f for f in os.listdir(dir) if f.endswith('.png')])
    print('Found {} files in directory'.format(png_files_num))

    path = dir + 'snapshot_'
    for i in range(0, png_files_num):
        img_path = path + str(i) + '.png'
        images.append(cv.imread(img_path, channels))

    return images


def display_image(image, size=(10, 10)):
    """
    Display the image given as a 2d or 3d array of values.
    :param size: Size of the plot for the image
    :param image: Input image to display
    """
    image = np.squeeze(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    fig = plt.figure(figsize=size)
    plt.imshow(image, interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    plt.savefig("test.png")
    plt.show()

# def display_image(image, size=(1000, 1000)):
#     """
#     Display the image given as a 2d or 3d array of values.
#     :param size: Size of the plot for the image
#     :param image: Input image to display
#     """
#     cv.namedWindow('image', cv.WINDOW_NORMAL)
#     cv.resizeWindow('image', size[0], size[1])
#     cv.imshow("image", image)
#     cv.waitKey(0)
#


def idf(img, ref_img):
    """
    Image Differencing Function RMSE
    :param img:
    :param ref_img:
    :return:
    """
    return math.sqrt(((ref_img - img)**2).mean())


def imgs_diff(ref_img, images):
    return np.array([[idf(ref_img, img)] for img in images])


def cov(a, b):
    """
    Calculates covariance (non sample)
    Assumes flattened arrays
    :param a:
    :param b:
    :return:
    """
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    return np.sum((a - a_mean) * (b - b_mean)) / (len(a))


def cor_coef(a, b):
    """
    Calculate correlation coefficient
    :param a:
    :param b:
    :return:
    """
    a = a.flatten()
    b = b.flatten()
    return cov(a, b) / (np.std(a) * np.std(b))


def imgs_coef(ref_img, images):
    return np.array([[cor_coef(ref_img, img)] for img in images])


def plot_line(data, label="Unknown", scatter=False):
    plt.plot(range(0, len(data)), data, label=label)
    if scatter: plt.scatter(range(0, len(data)), data)
    plt.show()


