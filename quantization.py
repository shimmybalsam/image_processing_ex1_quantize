import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
# from imageio import imread, imwrite
NUM_OF_BINS = 256
MAX_PIX_LEVEL = 255
GRAYSCALE = 1
RGB_DIM = 3

conversionMatrix = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])

def inverse(matrix):
    """
    helper function, calculates inversed matrix
    :param matrix: matrix
    :return: matrix inversed
    """
    return np.linalg.inv(matrix)

def read_image(filename, representation):
    """
    opens image as matrix
    :param filename: name of image
    :param representation: 1 if grayscale, 2 if RGB
    :return: np.array float64 of given image
    """
    image = imread(filename)
    image_float = image.astype(np.float64) / MAX_PIX_LEVEL
    if representation == GRAYSCALE:
        image_float_gray = rgb2gray(image_float)
        return image_float_gray
    return image_float


def imdisplay(filename, representation):
    """
    shows given image plot
    :param filename: name of image
    :param representation: 1 if grayscale, 2 if RGB
    :return: None
    """
    image = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    """
    converts rgb parameters to yiq
    :param imRGB: given rgb image
    :return: same image as yiq parameters
    """
    return np.dot(imRGB,conversionMatrix.transpose()).clip(0,1)

def yiq2rgb(imYIQ):
    """
    converts yiq parameters to rgb
    :param imYIQ: given yiq image
    :return: same image as rgb parameters
    """
    return np.dot(imYIQ, inverse(conversionMatrix).transpose()).clip(0,1)


def histogram_equalize(im_orig):
    """
    calculates image's histogram, and stretches\equalizes if needed
    :param im_orig: image
    :return: list of: equalized image, histogram of original image, histogram of equalized image.
    """
    if im_orig.ndim == RGB_DIM:
        im_yiq = rgb2yiq(im_orig)
        im_y = im_yiq[:,:,0]
        is_rgb = True
    else:
        im_y = im_orig
        is_rgb = False

    #unnormalize
    im_y = im_y * MAX_PIX_LEVEL
    im_y = im_y.astype(np.uint8)

    hist_orig, bounds = np.histogram(im_y, NUM_OF_BINS, (0, 255))
    cumulative = hist_orig.cumsum()
    normalized_cum = (cumulative * (MAX_PIX_LEVEL)) / im_y.size
    firstnonzero = normalized_cum[np.nonzero(normalized_cum)[0][0]]

    # look up table
    T_k = (normalized_cum-firstnonzero)/(normalized_cum[MAX_PIX_LEVEL]-firstnonzero) * MAX_PIX_LEVEL
    T_k = T_k.astype(np.uint8)
    new_y = T_k[im_y]

    #normalize
    new_y = new_y.astype(np.float64)
    new_y = new_y / MAX_PIX_LEVEL

    hist_eq, bounds_eq = np.histogram(new_y, NUM_OF_BINS, (0, 1))

    if is_rgb:
        im_yiq[:,:,0] = new_y
        im_eq = yiq2rgb(im_yiq)
    else:
        im_eq = new_y

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    quantizes the image
    :param im_orig: image
    :param n_quant: number of quant-bins
    :param n_iter: max amount of quantization iterations
    :return: list of: quantized image and error plot
    """
    if im_orig.ndim == RGB_DIM:
        im_yiq = rgb2yiq(im_orig)
        im_y = im_yiq[:,:,0]
        is_rgb = True
    else:
        im_y = im_orig
        is_rgb = False

    z = np.empty(n_quant + 1)
    q = np.empty(n_quant)
    error = []

    #unnormalize
    im_y = im_y * MAX_PIX_LEVEL
    im_y = im_y.astype(np.uint8)

    hist, bounds = np.histogram(im_y, NUM_OF_BINS, (0, 255))
    cumulative = hist.cumsum()
    num_of_pixels = cumulative[-1]

    # creation of initial z's
    z = np.array([np.abs(cumulative - i * num_of_pixels / n_quant).argmin()
                  for i in range(n_quant)] + [len(cumulative)])
    np.around(z)

    # quantization loop
    for n in range(n_iter):
        # creation of q's
        multiplied_hist_intesnity = np.arange(NUM_OF_BINS) * hist
        for j in range(n_quant):
            sum1 = np.sum(multiplied_hist_intesnity[int(np.ceil(z[j])):int(np.floor(z[j+1]) +1)])
            sum2 = np.sum(hist[int(np.ceil(z[j])):int(np.floor(z[j+1]) +1)])
            q[j] = (sum1 / sum2)
            np.ceil(q[j])

        # creation of new z's
        z[1:-1] = ((q[:-1] + q[1:]) / 2).astype(np.float64)
        np.around(z)

        #error calculation
        er_sum = 0
        for j in range(n_quant):
            for i in range(z[j], z[j + 1]):
                er_sum += ((q[j] - i)**2) * hist[i]

        # check if error converges
        if len(error) > 0 and error[-1] == er_sum:
            break
        else:
            error.append(er_sum)

    # look up table changing pixels to q's values
    map = np.zeros(NUM_OF_BINS)
    for j in range(n_quant):
        map[z[j]:z[j+1]] = q[j]
    new_y = map[im_y]

    #normalize
    new_y = new_y.astype(np.float64)
    new_y = new_y / MAX_PIX_LEVEL

    if is_rgb:
        im_yiq[:,:,0] = new_y
        im_quant = yiq2rgb(im_yiq)
    else:
        im_quant = new_y

    return [im_quant, error]

im = read_image("images\jerusalem.jpg",2)
plt.imshow(im)
plt.show()
plt.imshow(histogram_equalize(im)[0])
plt.show()
a,b = quantize(im,3,10)
plt.imshow(a)
plt.show()
plt.plot(b)
plt.show()


