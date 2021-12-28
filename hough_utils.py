from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import pdb
import cv2

DEFAULT_HEIGHT = 640
DEFAULT_GAUSSIAN_KERNEL_SIZE = (11, 11)
DEFAULT_GAUSSIAN_KERNEL_VARIANCE = 1.5

def preprocess(image):
    '''Takes in a pillow image and resizes it to a fixed height preserving aspect ratio

    Args:
        image (Pillow image): An image that has to be preprocessed
    
    Returns:
        Pillow image: A resized image
    
    NOTE:
        It does not resize images smaller than the DEFAULT_HEIGHT
    '''
    h = image.height
    w = image.width
    height = DEFAULT_HEIGHT
    
    # resize the image only if the height is greater than the default height
    if h > height:
        width = int(w * (height/h))
        image = image.resize((width, height), Image.ANTIALIAS)
    return image

def gaussianFilter(image):
    return cv2.GaussianBlur(image, ksize = DEFAULT_GAUSSIAN_KERNEL_SIZE, sigmaX=DEFAULT_GAUSSIAN_KERNEL_VARIANCE)

def sobelFilter(image):
    '''Finds gradients along the X and Y axis 
    
    Args: 
        image (ndarray): An ndarray corresponding to the input image

    Returns:
        ndarray: An ndarray containing gradients corresponding to the X and Y axis.
    '''
    # Grad Y
    sobel_x = cv2.Sobel(image,cv2.CV_64F,1,0, ksize=3)

    # Grad X
    sobel_y = cv2.Sobel(image,cv2.CV_64F,0,1, ksize=3)

    # Find the L2 of X Gradient and Y Gradient
    return np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))

def rhoToRow(rho_predicted, rho_delta, rho_max):
    '''Convert rho (normal length) to the corresponding row of the accumulator
    
    Args:
        rho_predicted (float): The normal length corresponding to the line
        rho_delta (float): The change in rho with each row

    Returns:
        (int) The row corresponding to the input rho
    '''
    offset = int((rho_predicted + np.abs(rho_max))/rho_delta)
    return offset

def rowToRho(row, rho_delta, rho_max):
    '''Convert the row in the acumulator array to the corresponding rho

    Args:
        row (int): Row in the accumulator array
        rho_delta (float): Change in rho by changing a single row
        rho_max (int): Maximum possible rho
    
    Returns:
        (float): rho chorresponding to the input row
    '''
    rho = -np.abs(rho_max) + row * rho_delta
    return rho

def selectTop(accumulator, top = 50):
    threshold = np.partition(accumulator.flatten(), -top)[-top]

    accumulator[accumulator < threshold] = 0
    return accumulator

def draw_inf_lines(image, accumulator, rho_max, rho_delta, length = 10000):
    '''Draw lines of infinite length for edges in the images
    
    Args:
        image (ndarray): An image on which we want to draw lines
        accumulator (ndarray): An array containing votes for lines
        rho_max (float): The maximal possible normal distance
        rho_delta (float): The change in rho by changing a single row
        length (float): The length of the line to be drawn on the image

    Returns:
        An image with lines drawn corresponding to lines detected from the image
    '''
    # convert to PIL Image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    #draw.line((-1.5,0.5, 100.333, 100.5), fill=255, width = 3)

    indices = np.where(accumulator>0)

    for i in range(len(indices[0])): 
        # get x0 y0
        rho = rowToRho(indices[0][i], rho_delta, rho_max)
        theta = indices[1][i]
        theta = (theta * np.pi)/180

        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)

        dx = - np.sin(theta)
        dy = np.cos(theta)

        # get x1 y1, x2 y2 from dx, dy
        x1 = int(x0 + dx * length)
        y1 = int(y0 + dy * length)

        x2 = int(x0 - dx * length)
        y2 = int(y0 - dy * length)

        #draw line
        draw.line((x1, y1, x2, y2), fill=255, width = 1)
    
    return image

def draw_finite_lines(image, accumulator, rho_max, rho_delta, voters, tolerence = 0.05, length = 5):
    '''Draw lines of finite length for edges in the images
    
    Args:
        image (ndarray): An image on which we want to draw lines
        accumulator (ndarray): An array containing votes for lines
        rho_max (float): The maximal possible normal distance
        rho_delta (float): The change in rho by changing a single row
        voters (ndarray): An array containing the pixels that voted for an edge
        tolerence (float): The amount of error we allow for rho_predicted
        length (float): The length of the line to be drawn on the image

    Returns:
        An image with lines drawn corresponding to lines detected from the image
    '''
    # convert to PIL Image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    indices = np.where(accumulator>0)
    num_voters = len(voters[0])

    for voter_index in tqdm(range(num_voters)):
        voter_y = voters[0][voter_index]
        voter_x = voters[1][voter_index]

        for i in range(len(indices[0])): 
            rho = rowToRho(indices[0][i], rho_delta, rho_max)
            theta = indices[1][i]
            theta = (theta * np.pi)/180

            rho_target = voter_x * np.cos(theta) + voter_y * np.sin(theta)
            diff = rho - rho_target

            if np.abs(diff) <= tolerence:
                dx = - np.sin(theta)
                dy = np.cos(theta)

                x1 = int(voter_x + dx * length)
                y1 = int(voter_y + dy * length)

                x2 = int(voter_x - dx * length)
                y2 = int(voter_y - dy * length)

                #draw line
                draw.line((x1, y1, x2, y2), fill="green", width = 1)
    
    return image

def render_image(image):
    image.show()

def isMax(image, x, y, window_size):
    '''Finds out if the pixel in the middle of the window is the largets

    Args:
        image (ndarray): An ndarray representing the image
        x (int): The x coordinate of the start of the window
        y (int): The y coordinate of the start of the window
        window_size (int): The size of the window in which we want to find max
    
    Returns:
        True if the center pixel is greater than all the other pixels in the window
    '''
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
    '''Retain the local maxima of votes

    Args:
        image (ndarray): The image represented as an ndarray
        window_size (int): The size of the window around which we want to find local maxima

    Returns:
        (ndarray): The image filtered for the local maxima
    '''
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