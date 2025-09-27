import cv2
import numpy as np


def image_segmentation(img):
    # img = median_filter(img)

    # алгоритм Кэнни
    # Вычисление градиента
    # Gray = 0.299 * R + 0.587 * G + 0.114 * B
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # один цветовой канал
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Бинаризация изображения градиента
    _, binary_image = cv2.threshold(
        np.uint8(grad_magnitude),
        100,
        255,
        cv2.THRESH_BINARY
    )  # бинарная (черно-белая) пороговая обработка

    # Подавление немаксимумов
    non_max_suppressed = np.zeros(binary_image.shape, dtype=np.uint8)
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            if binary_image[i, j] == 255:
                dx = grad_x[i, j]
                dy = grad_y[i, j]
                magnitude = grad_magnitude[i, j]

                if (dx != 0) or (dy != 0):
                    angle = np.arctan2(dy, dx)
                    angle_deg = angle * 180. / np.pi

                    if 0 <= angle_deg < 22.5 or 157.5 <= angle_deg <= 180:
                        # Горизонтальное направление
                        if (magnitude >= grad_magnitude[i, j - 1]) and (magnitude >= grad_magnitude[i, j + 1]):
                            non_max_suppressed[i, j] = magnitude
                    elif 22.5 <= angle_deg < 67.5:
                        # Диагональное 1 (верхний левый - нижний правый)
                        if (magnitude >= grad_magnitude[i - 1, j - 1]) and (
                                magnitude >= grad_magnitude[i + 1, j + 1]):
                            non_max_suppressed[i, j] = magnitude
                    elif 67.5 <= angle_deg < 112.5:
                        # Вертикальное направление
                        if (magnitude >= grad_magnitude[i - 1, j]) and (magnitude >= grad_magnitude[i + 1, j]):
                            non_max_suppressed[i, j] = magnitude
                    else:
                        # Диагональное 2 (верхний правый - нижний левый)
                        if (magnitude >= grad_magnitude[i - 1, j + 1]) and (
                                magnitude >= grad_magnitude[i + 1, j - 1]):
                            non_max_suppressed[i, j] = magnitude
    return non_max_suppressed


if __name__ == '__main__':
    # Загрузка изображения
    img = cv2.imread('lab4_images/shrek.jpg')

    result = image_segmentation(img)

    # Сохранение результата
    cv2.imwrite('lab4_results/shrek_2.png', result)