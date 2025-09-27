import cv2
import numpy as np

# choose the image file
image = cv2.imread('images/shrek.jpg')

# get image width and height
height, width = image.shape[:2]
image = cv2.resize(image, (int(width/3), int(height/3)),
                   interpolation=cv2.INTER_AREA)

# pencil filter and sketch
pencil, sketch = cv2.pencilSketch(
    image, sigma_s=40, sigma_r=0.06, shade_factor=0.05)

# place text at center
def place_text(text, image):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# place text on the images
# place_text('Original image', image)
# place_text('Pencil image', pencil)
# place_text('Sketch image', sketch)

# sepia filter
sepia = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]])
sepia_image = cv2.transform(image, sepia)
# place_text('Sepia image', sepia_image)

# greyscale filter
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Convert grayscale to 3-channel image for text placement
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# place_text('Grayscale image', gray_bgr)

# Create another filter for the 6th image (invert colors)
inverted = cv2.bitwise_not(image)
# place_text('Inverted image', inverted)

# show all 6 images
cv2.imshow('1 - Original image', image)
cv2.imshow('2 - Pencil image', pencil)
cv2.imshow('3 - Sketch image', sketch)
cv2.imshow('4 - Sepia image', sepia_image)
cv2.imshow('5 - Grayscale image', gray_bgr)
cv2.imshow('6 - Inverted image', inverted)

# wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()