import cv2
import numpy as np

image = cv2.imread('images/shrek.jpg')

height, width = image.shape[:2]
image = cv2.resize(image, (int(width / 3), int(height / 3)),
                   interpolation=cv2.INTER_AREA)

pencil, sketch = cv2.pencilSketch(
    image, sigma_s=40, sigma_r=0.06, shade_factor=0.05)

# Создание сепия фильтра с помощью матрицы преобразования цветов
sepia = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]])
sepia_image = cv2.transform(image, sepia)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

inverted = cv2.bitwise_not(image)


def add_label(img, text, height_extra=40):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    labeled = cv2.copyMakeBorder(img, 0, height_extra, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(50, 50, 50))

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (labeled.shape[1] - text_size[0]) // 2
    text_y = img.shape[0] + (height_extra + text_size[1]) // 2 - 5
    cv2.putText(labeled, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return labeled


img1 = add_label(image, "Original")
img2 = add_label(pencil, "Pencil")
img3 = add_label(sketch, "Sketch")
img4 = add_label(sepia_image, "Sepia")
img5 = add_label(gray_bgr, "Grayscale")
img6 = add_label(inverted, "Inverted")

row1 = np.hstack([img1, img2, img3])
row2 = np.hstack([img4, img5, img6])
collage = np.vstack([row1, row2])

cv2.imshow("All Filters", collage)

cv2.waitKey(0)
cv2.destroyAllWindows()
