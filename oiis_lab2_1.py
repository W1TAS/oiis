from PIL import Image, ImageDraw


def med(image, x, y, z):
    pix = image.load()
    window_values = []
    for i in range(-1, 2):
        for j in range(-1, 2):  # диапазон 11×11 пикселей
            if (x+i >= 0) and (y+j >= 0) and (x+i < image.size[0]) and (y+j < image.size[1]):
                window_values.append(pix[x+i, y+j][z])
    window_values.sort()
    middle_index = len(window_values) // 2  # Настоящая медиана
    return window_values[middle_index]


image = Image.open(r"images\shrek.jpg")

# Используем правильный режим "RGB" вместо "JPEG"
new = Image.new("RGB", image.size)
draw = ImageDraw.Draw(new)

for i in range(image.size[0]):
    for j in range(image.size[1]):
        r = med(image, i, j, 0)
        g = med(image, i, j, 1)
        b = med(image, i, j, 2)
        draw.point((i, j), (r, g, b))

# Сохраняем как JPEG
new.save(r"images\shrek_smooth.jpg", "JPEG")