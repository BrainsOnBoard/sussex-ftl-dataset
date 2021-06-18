import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
import pandas as pd
import pathlib
from collections.abc import Iterable

def load_images(dir, greyscale=False):
    '''
    Read the images from the given directory
    :param dir:
    :param greyscale:
    :return: List of nd.Array images
    '''
    if greyscale:
        channels = cv.IMREAD_GRAYSCALE
    else:
        channels = cv.IMREAD_COLOR

    images = []
    # count .png files
    png_files_num = len([f for f in os.listdir(dir) if f.endswith('.png')])
    print('Found {} files in directory: {}'.format(png_files_num, dir))
    path = os.path.join(dir, 'training.csv')
    data = pd.read_csv(path).to_dict('list')
    for i in data[' Filename']:
        img_path = os.path.join(dir, i.strip("/").split('/')[-1])
        images.append(cv.imread(img_path, channels))
    print('Loaded {} images with shape {}'.format(len(images), images[0].shape))
    return images


def display_image(image, size=(10, 10), save=False):
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
    if save: plt.savefig("image.png")
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
    return cv.absdiff(ref_img, img).mean()


def heatmap(ref_img, img):
    heat = cv.absdiff(ref_img, img)
    plt.imshow(heat, cmap='hot', interpolation='nearest')
    plt.show()


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


def plot_multiline(data, scatter=False):
    for line in data:
        plt.plot(range(0, len(line)), line)
        if scatter: plt.scatter(range(0, len(line)), line)
    plt.show()


def rotate(d, image):
    """
    Converts the degrees into columns and rotates the image.
    Positive degrees rotate the image clockwise
    and negative degrees rotate the image counter clockwise
    :param d: number of degrees the agent will rotate its view
    :param image: An np.array that we want to shift.
    :return: Returns the rotated image.
    """
    assert abs(d) <= 360

    num_of_cols = image.shape[1]
    num_of_cols_perdegree = num_of_cols / 360
    cols_to_shift = round(d * num_of_cols_perdegree)
    return np.roll(image, -cols_to_shift, axis=1)


def ridf(ref_img, current_img,  degrees=306, step=1):
    degrees = round(degrees/2)
    rmse = []   # Hold the RMSEs between the current and the image of the route for every degree
    for k in range(-degrees, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        rmse.append(idf(curr_image, ref_img))  #IDF function to find the error between the selected route image and the rotated current
    return rmse


def r_cor_coef(ref_img, current_img,  degrees, step):
    '''
    Calculates rotational correlation coefficients
    :param ref_img:
    :param current_img:
    :param degrees:
    :param step:
    :return:
    '''
    degrees = round(degrees/2)  # degrees to rotate for left and right
    r_coef = []   # Hold the r_coefs between the current and the image of the route for every degree
    for k in range(-degrees, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        # coe_coef function to find the correlation between the selected route image and the rotated current
        r_coef.append(cor_coef(curr_image, ref_img))
    return r_coef


def tridf(ref_img, route_imgs, degrees=360, step=1):
    '''
    Translation RIDF. Calculates the rotation image difference
    between a given reference image and the rest of the given
    image dataset.
    :param ref_img:
    :param route_imgs:
    :param degrees:
    :param step:
    :return:
    '''
    trmse = []
    for img in route_imgs:
        ridf_logs = ridf(ref_img, img, degrees, step)
        trmse.append(ridf_logs)
    return np.array(trmse)


def ridf_field(route_imgs, degrees, step):
    '''
    RIDF field. Calcualtes the rotation image difference (ridf)
    between every image in the dataset and itself
    :param route_imgs:
    :param degrees:
    :param step:
    :return:
    '''
    field = []
    for img in route_imgs:
        ridf_logs = ridf(img, img, degrees, step)
        field.append(ridf_logs)
    return np.array(field)


def cor_coef_field(route_imgs, degrees, step):
    '''
    Correlation Coefficient field. Calcualtes the rotational correlation coefficient
    between every image in the dataset and itself
    :param route_imgs:
    :param degrees:
    :param step:
    :return:
    '''
    coef = []
    for img in route_imgs:
        coef_logs = r_cor_coef(img, img, degrees, step)
        coef.append(coef_logs)
    return np.array(coef)


def crop(imgs, x=0, y=0, h=-1, w=-1):
    if isinstance(imgs, Iterable):
        return [i[y:y + h, x:x + w] for i in imgs]
    else:
        return imgs[y:y + h, x:x + w]


def plot_3d(data, show=True, rows_cols_idx=111, title=''):
    '''
    Plots the 2d data given in a 3d wireframe.
    Assumes first dimension is number of images,
    second dimension is search angle.
    :param data:
    :return:
    '''
    # The second dimension of the data is the search angle
    # i.e the degree rotated to the left (-deg) and degree rotated to the right (+deg)
    deg = round(data.shape[1]/2)
    no_of_imgs = data.shape[0]

    x = np.linspace(-deg, deg, deg*2)
    y = np.linspace(0, no_of_imgs, no_of_imgs)
    X, Y = np.meshgrid(x, y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.subplot(rows_cols_idx, projection='3d')
    ax.plot_wireframe(X, Y, data)
    ax.title.set_text(title)
    if show: plt.show()
