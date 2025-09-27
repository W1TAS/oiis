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




def generate_signal(function, frequency, length):
    t = np.linspace(0, 1, length, endpoint=False)
    if function == 'sin':
        signal = np.sin(2 * np.pi * frequency * t) # f(t) = A sin(2πft + φ), где φ = 0, A = 1
    elif function == 'cos':
        signal = np.cos(2 * np.pi * frequency * t) # f(t) = A cos(2πft + φ), где φ = 0, A = 1
    else:
        raise ValueError("Выберите либо 'sin', либо 'cos'")
    return signal


function = input("Введите функцию (sin или cos): ")
frequency = int(input("Введите частоту: "))
length = 235

signal = generate_signal(function, frequency, length)
next_pow_of_2 = 1 << (length - 1).bit_length()
signal = list(signal)
signal = signal + [0] * (next_pow_of_2 - length)

# Тестирование FFT и сравнение с NumPy FFT
signal_after_FFT = FFT(signal)  # Используем исправленную функцию FFT
signal_after_fft = np.fft.fft(signal)

# Построение графиков
plt.figure(figsize=(10, 9))

plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title("Исходный сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

plt.subplot(3, 1, 2)
plt.plot(np.abs(signal_after_FFT))
plt.title("Спектр сигнала (Реализованный БПФ)")
plt.xlabel("Частота")
plt.ylabel("Амплитуда")

plt.subplot(3, 1, 3)
plt.plot(np.abs(signal_after_fft))
plt.title("Спектр сигнала (NumPy БПФ)")
plt.xlabel("Частота")
plt.ylabel("Амплитуда")

plt.tight_layout()
plt.show()