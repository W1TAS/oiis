import cv2
import numpy as np


def smoothing(img):
    """Сглаживание (размытие Гаусса для удаления шума)"""
    return cv2.GaussianBlur(img, (5, 5), 1.4)


def compute_gradients(img):
    """поиск градиентов оператором Собеля"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # сила изменения яркости слева-направо
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # сила изменения яркости сверху-вниз

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)  # Насколько резко меняется яркость
    grad_direction = np.arctan2(grad_y, grad_x)  # В какую сторону яркость меняется сильнее всего

    return grad_magnitude, grad_direction


def non_maximum_suppression(grad_magnitude, grad_direction):
    """Подавление немаксимумов."""
    rows, cols = grad_magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.uint8)

    angle = grad_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            # направление градиента(Для каждого направления сравниваем с соседями)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):  # горизонтальное направление
                q = grad_magnitude[i, j + 1]
                r = grad_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:  # диагональ ↘
                q = grad_magnitude[i + 1, j - 1]
                r = grad_magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:  # вертикальное направление
                q = grad_magnitude[i + 1, j]
                r = grad_magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:  # диагональ ↖
                q = grad_magnitude[i - 1, j - 1]
                r = grad_magnitude[i + 1, j + 1]

            # Если текущий пиксель сильнее обоих соседей вдоль границы → оставляем его, иначе → удаляем.
            if (grad_magnitude[i, j] >= q) and (grad_magnitude[i, j] >= r):  # диагональ ↖
                suppressed[i, j] = grad_magnitude[i, j]

    return suppressed


def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    """Двойная пороговая фильтрация (Потенциальные границы определяются порогами)"""

    # Вычисляем верхний и нижний пороги на основе максимального значения в изображении
    high_threshold = img.max() * high_ratio  # Верхний порог: 15% от максимума
    low_threshold = high_threshold * low_ratio  # Нижний порог: 5% от верхнего порога

    res = np.zeros(img.shape, dtype=np.uint8)

    strong = 255  # сильная граница
    weak = 75  # слабая граница (проверка является ли границей в def hysteresis)

    # Находим координаты сильных границ - пиксели со значением ВЫШЕ верхнего порога
    strong_i, strong_j = np.where(img >= high_threshold)
    # Находим координаты слабых границ - пиксели со значением МЕЖДУ порогами
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    # Отмечаем сильные границы белым цветом
    res[strong_i, strong_j] = strong
    # Отмечаем сильные границы белым цветом
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
    """Трассировка области неоднозначности (связывание слабых границ с сильными)."""
    rows, cols = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img[i, j] == weak:
                # проверяем всех соседей вокруг слабого пикселя
                if np.any(img[i - 1:i + 2,
                          j - 1:j + 2] == strong):  # если хотябы один сосед сильный, то пиксель становится сильным
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def image_segmentation(img):
    img = smoothing(img)

    grad_magnitude, grad_direction = compute_gradients(img)

    suppressed = non_maximum_suppression(grad_magnitude, grad_direction)

    thresholded, weak, strong = double_threshold(suppressed)

    result = hysteresis(thresholded, weak, strong)

    return result


if __name__ == '__main__':
    img = cv2.imread('lab4_images/shrek.jpg')
    result = image_segmentation(img)
    cv2.imwrite('lab4_results/shrek_canny.png', result)
