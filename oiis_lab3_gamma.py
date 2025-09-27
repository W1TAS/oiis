import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    # Простая и эффективная гамма-коррекция
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Пример использования
img = cv2.imread('lab4_images/1.png')
img_corrected = gamma_correction(img, gamma=0.5)  # gamma < 1 для тёмных изображений
cv2.imshow('img', img)
cv2.imshow('img_corrected', img_corrected)
# wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()