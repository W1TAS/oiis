import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('lab4_images/1.png').convert('L')
img_array = np.asarray(img)


# метод гистограммного выравнивания (делаем гистограмму равномерной)
def histogram_equalization(image):
    flat = image.flatten()

    histogram = np.zeros(256)
    # Заполняем гистограмму: подсчитываем количество пикселей каждой яркости
    for pixel in flat:
        histogram[pixel] += 1
    # функция распределения
    cdf = histogram.cumsum()

    # Формула линейно растягивает значения CDF на весь диапазон 0-255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    equalized = cdf_normalized[flat]
    equalized_image = np.reshape(equalized, image.shape)

    return equalized_image


# Применяем выравнивание гистограммы
equalized_img = histogram_equalization(img_array)

# Отображение результатов
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title('После выравнивания гистограммы')
plt.axis('off')

# для гистограмм
# ось X (горизонтальная) — уровни яркости от 0 (чёрный) до 255 (белый),
# ось Y (вертикальная) — количество пикселей с данным значением яркости.

plt.subplot(2, 2, 3)
plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='gray')
plt.title('Гистограмма исходного изображения')

plt.subplot(2, 2, 4)
plt.hist(equalized_img.flatten(), bins=256, range=(0, 256), color='gray')
plt.title('Гистограмма после выравнивания')

plt.tight_layout()
plt.show()
