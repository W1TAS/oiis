import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('lab4_images/1.png')
img_array = np.asarray(img)


def histogram_equalization(image):
    # Преобразуем в одномерный массив
    flat = image.flatten()

    # Создаем гистограмму
    histogram = np.zeros(256)
    for pixel in flat:
        histogram[pixel] += 1

    # Вычисляем кумулятивную функцию распределения (CDF)
    cdf = histogram.cumsum()

    # Нормализуем CDF к диапазону 0-255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # Применяем преобразование к изображению
    equalized = cdf_normalized[flat]
    equalized_image = np.reshape(equalized, image.shape)

    return equalized_image


# Применяем выравнивание гистограммы
equalized_img = histogram_equalization(img_array)

# Отображение результатов
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title('После выравнивания гистограммы')
plt.axis('off')

plt.tight_layout()
plt.show()
