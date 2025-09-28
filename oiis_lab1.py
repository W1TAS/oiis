import numpy as np
import cmath
import matplotlib.pyplot as plt


def FFT(P):
    n = len(P)  # n is a power of 2

    if n == 1:
        return P

    w = cmath.exp(2j * cmath.pi / n)  # e^(2πi/n)

    P_e = P[0::2]  # Четные коэффициенты [p0, p2, p4, ...]
    P_o = P[1::2]  # Нечетные коэффициенты [p1, p3, p5, ...]

    y_e = FFT(P_e)  # Рекурсивный вызов для четных
    y_o = FFT(P_o)  # Рекурсивный вызов для нечетных

    y = [0] * n  # Инициализация результата

    for j in range(n // 2):
        y[j] = y_e[j] + (w ** j) * y_o[j]
        y[j + n // 2] = y_e[j] - (w ** j) * y_o[j]
    return y


def generate_signal(function, frequency, num_points, duration):
    """
    Генерация сигнала

    Parameters:
    function: 'sin' или 'cos'
    frequency: частота сигнала (Гц)
    num_points: количество точек дискретизации
    duration: длительность временной линии (секунды)
    """
    t = np.linspace(0, duration, num_points, endpoint=False)
    if function == 'sin':
        signal = np.sin(2 * np.pi * frequency * t)  # f(t) = A sin(2πft + φ), где φ = 0, A = 1
    elif function == 'cos':
        signal = np.cos(2 * np.pi * frequency * t)  # f(t) = A cos(2πft + φ), где φ = 0, A = 1
    else:
        raise ValueError("Выберите либо 'sin', либо 'cos'")
    return t, signal


# Ввод параметров
function = input("Введите функцию (sin или cos): ")
frequency = int(input("Введите частоту (Гц): "))
num_points = int(input("Введите количество точек: "))
duration = float(input("Введите длительность временной линии (секунды): "))

# Генерация сигнала
t, signal = generate_signal(function, frequency, num_points, duration)

# Дополнение нулями до степени двойки
next_pow_of_2 = 1 << (num_points - 1).bit_length()
signal_padded = list(signal)
signal_padded = signal_padded + [0] * (next_pow_of_2 - num_points)

# Тестирование FFT и сравнение с NumPy FFT
signal_after_FFT = FFT(signal_padded)  # Используем исправленную функцию FFT
signal_after_fft = np.fft.fft(signal_padded)

# Расчет частот для оси X спектра
sampling_rate = 1024  # частота дискретизации
freq_axis = np.fft.fftfreq(len(signal_padded), 1 / sampling_rate)

# Построение графиков
plt.figure(figsize=(12, 10))

# Исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title(f"Исходный сигнал: {function}(2π*{frequency}*t)")
plt.xlabel("Время (секунды)")
plt.ylabel("Амплитуда")
plt.grid(True)

# Спектр (реализованный БПФ)
plt.subplot(3, 1, 2)
plt.plot(freq_axis[:len(freq_axis) // 2], np.abs(signal_after_FFT)[:len(signal_after_FFT) // 2])
plt.title("Спектр сигнала (Реализованный БПФ)")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.grid(True)

# Спектр (NumPy БПФ)
plt.subplot(3, 1, 3)
plt.plot(freq_axis[:len(freq_axis) // 2], np.abs(signal_after_fft)[:len(signal_after_fft) // 2])
plt.title("Спектр сигнала (NumPy БПФ)")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод информации о сигнале
print(f"\nИнформация о сигнале:")
print(f"Количество точек: {num_points}")
print(f"Длительность: {duration} секунд")
print(f"Частота дискретизации: {sampling_rate:.2f} Гц")
print(f"Дополнено до {next_pow_of_2} точек (ближайшая степень двойки)")