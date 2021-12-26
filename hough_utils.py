import numpy as np
from PIL import Image, ImageDraw
import pdb
import cv2

def preprocess(image):
    h = image.height
    w = image.width
    height = 640
    width = int(w * (height/h))
    image = image.resize((width, height), Image.ANTIALIAS)
    return image

def gaussianFilter(image):
    return cv2.GaussianBlur(image, ksize = (11,11), sigmaX=1.5)

def sobelFilter(image):
    sobel_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    return np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))

def rhoToRow(rho_predicted, rho_delta, rho_max):
    offset = int((rho_predicted + np.abs(rho_max))/rho_delta)
    return offset

def rowToRho(row, rho_delta, rho_max):
    rho = -np.abs(rho_max) + row * rho_delta
    return rho

def selectTop(accumulator, top = 50):
    threshold = np.partition(accumulator.flatten(), -top)[-top]

    accumulator[accumulator < threshold] = 0
    return accumulator

def draw_inf_lines(image, accumulator, rho_max, rho_delta, length = 10000):
    # conver to PIL Image
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

        #pdb.set_trace()

        #draw line
        draw.line((x1, y1, x2, y2), fill=255, width = 1)
    
    return image



def render_image(image):
    image.show()

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