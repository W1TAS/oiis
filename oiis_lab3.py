from PIL import Image
import numpy as np

img1 = Image.open('images/shrek.jpg')
img2 = Image.open('images/night.jpg')

img1_arr = np.array(img1, dtype=np.float32)
img2_arr = np.array(img2, dtype=np.float32)

img1_mean = np.mean(img1_arr)
img2_mean = np.mean(img2_arr)

target = (img1_mean + img2_mean) / 2


# Простой способ: перемещаем среднее значение к цели
def adjust_brightness(image, current_mean, target_mean):
    # Находим насколько нужно изменить среднее
    mean_diff = target_mean - current_mean

    # Применяем изменение ко всем пикселям
    # Но с нелинейным эффектом: тёмные осветляем сильнее, светлые - слабее
    adjusted = image + mean_diff * (1 - (image - 128) / 384)  # Нелинейная коррекция
    return np.clip(adjusted, 0, 255)


new_img1_arr = adjust_brightness(img1_arr, img1_mean, target)
new_img2_arr = adjust_brightness(img2_arr, img2_mean, target)

Image.fromarray(new_img1_arr.astype(np.uint8)).save('images/shrek_r2.jpg')
Image.fromarray(new_img2_arr.astype(np.uint8)).save('images/night_r2.jpg')