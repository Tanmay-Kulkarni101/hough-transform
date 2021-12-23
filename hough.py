import numpy as np
import hough_utils as utils
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

MAX_DEGREE = 180
THETA_OFFSET = 0

def hough(image, rho_delta = 1, theta_delta = 1, rho_max = 500, threshold = 255.0/2, debug = False):
    filtered_image = utils.sobelFilter(image)

    indices = np.where(filtered_image > threshold)
    
    rows = 2 * int(rho_max/rho_delta)
    cols = int(MAX_DEGREE/theta_delta)
    
    accumulator = np.zeros((rows, cols))

    for i in range(len(indices[0])):
        x = indices[0][i]
        y = indices[1][i]

        for col in range(cols):
            theta = (THETA_OFFSET + col * theta_delta * np.pi)/180

            rho_predicted = y * np.cos(theta) + x * np.sin(theta)

            if np.abs(rho_predicted) >= rho_max:
                rho_predicted = (rho_predicted/np.abs(rho_predicted)) * rho_max

            corresponding_row = utils.rhoToRow(rho_predicted, rho_delta, rho_max)

            if corresponding_row >= rows:
                corresponding_row = rows - 1
            
            #print(corresponding_row, col)
            if corresponding_row < 0:
                print(x, y, rho_predicted, corresponding_row, theta)
            accumulator[corresponding_row,col] += 1
    
    # if debug:
    #     plt.imshow(accumulator, cmap='hot', interpolation='nearest')
    #     plt.show()
    
    accumulator = utils.non_max_suppression(accumulator)

    if debug:
        plt.imshow(accumulator, cmap='hot', interpolation='nearest')
        plt.show()

    return accumulator

if __name__ == '__main__':
    image = Image.open('./images/Unknown.jpeg') # get this from the upload
    image = ImageOps.grayscale(image)
    #image.show()

    image = np.asarray(image)
    #print(image)
    #image = utils.preprocess(image)

    accumulator = hough(image)
    image_with_lines = utils.draw_lines(image, accumulator)

    utils.render_image(image_with_lines)