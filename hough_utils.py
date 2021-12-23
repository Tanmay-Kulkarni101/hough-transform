from scipy import ndimage
import numpy as np
from PIL import Image

def preprocess(image):
    pass

def sobelFilter(image):
    return ndimage.sobel(image)

def draw_lines(image, rhos_thetas_list):
    pass

def render_image(image):
    pass

def isMax(image, x, y, window_size):
    center_x = x + int(window_size/2)
    center_y = y + int(window_size/2)

    for i in range(window_size):
        for j in range(window_size):
            curr_x = x + i
            curr_y = y + j
            
            if curr_x == center_x and curr_y == center_y:
                continue

            if image[curr_x, curr_y] >= image[center_x, center_y]:
                return center_x, center_y, False

    return center_x, center_y, True

def non_max_suppression(image, window_size = 3):
    padding = int(window_size/2)
    padded_image = np.pad(image, pad_width = padding)

    result = np.zeros_like(padded_image)

    rows, cols = padded_image.shape

    # we are indexing from the left corner of the window_size
    # rows - 1 >= i + window_size - 1
    for i in range(rows - window_size+1):
        for j in range(cols - window_size+1):
            x , y, isMaxVal = isMax(padded_image, i, j, window_size)
            result[x,y] = padded_image[x,y] if isMaxVal else 0

    # getting rid of padding
    return result[padding:-padding,padding:-padding]

def rhoToRow(rho_predicted, rho_delta, rho_max):
    offset = int((rho_predicted + np.abs(rho_max))/rho_delta)
    return offset
