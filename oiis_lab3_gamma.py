

import cv2
import numpy as np


def gamma_correction(img_path, gamma):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Cannot load image {img_path}")
        return None
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return gamma_corrected


# Обрабатываем изображение 1.png с gamma=2.2 (стандартное значение)
if __name__ == '__main__':
    input_image = 'lab4_images/1.png'
    gamma_value = 0.5

    result = gamma_correction(input_image, gamma_value)

    if result is not None:
        cv2.imwrite('1_corrected.png', result)
        print('Gamma correction completed! Saved as 1_corrected.png')
    else:
        print('Error processing image')

