import numpy as np
import hough_utils as utils
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm

MAX_DEGREE = 180
THETA_OFFSET = 0

def hough(image, rho_delta = 1, theta_delta = 1, threshold = 30.0, debug = False):
    h,w = image.shape

    # Largest possible normal distance is the diagonal length of the image
    rho_max = np.linalg.norm(np.array([h,w]))

    # Obtain edges corresponding to the image
    image = utils.gaussianFilter(image)
    filtered_image = utils.sobelFilter(image)

    indices = np.where(filtered_image > threshold)
    
    rows = 2 * int(rho_max/rho_delta)
    cols = int(MAX_DEGREE/theta_delta)
    
    accumulator = np.zeros((rows, cols))

    for i in tqdm(range(len(indices[0]))):
        y = indices[0][i]
        x = indices[1][i]

        for col in range(cols):
            theta = (THETA_OFFSET + col * theta_delta * np.pi)/180

            rho_predicted = x * np.cos(theta) + y * np.sin(theta)

            if np.abs(rho_predicted) >= rho_max:
                rho_predicted = (rho_predicted/np.abs(rho_predicted)) * rho_max

            corresponding_row = utils.rhoToRow(rho_predicted, rho_delta, rho_max)

            if corresponding_row >= rows:
                corresponding_row = rows - 1
            
            accumulator[corresponding_row,col] += 1
    
    if debug:
        plt.imshow(accumulator, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
    
    accumulator = utils.non_max_suppression(accumulator)

    if debug:
        plt.imshow(accumulator, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return accumulator, rho_max, indices

if __name__ == '__main__':
    image = Image.open('./images/Unknown.jpeg') # get this from the upload
    grey_scale_image = ImageOps.grayscale(image)
    image = utils.preprocess(image)

    grey_scale_image = np.asarray(grey_scale_image)
    image = np.asarray(image)
    
    rho_delta = 1
    accumulator, rho_max, voters = hough(grey_scale_image, rho_delta=rho_delta)
    accumulator = utils.selectTop(accumulator)

    image_with_lines = utils.draw_finite_lines(image, accumulator, rho_max, rho_delta, voters)
    # image_with_lines = utils.draw_inf_lines(image, accumulator, rho_max, rho_delta)

    utils.render_image(image_with_lines)