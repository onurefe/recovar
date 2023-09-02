import numpy as np
import matplotlib.pyplot as plt


def unit_box(f, T):
    return T * np.sinc(f * T)


t1 = np.linspace(start=-30, stop=60, endpoint=False, num=9000)
t2 = np.linspace(start=0, stop=30, endpoint=False, num=3000)

f = np.fft.fftfreq(len(t1), 0.01)
T = 10

x = np.tanh(t2)
xp = np.pad(x, (3000, 3000))
xw = np.fft.fft(xp)
yw = unit_box(f, T)
y1 = np.fft.ifft(xw * yw)
y2 = np.fft.ifft(xw)

plt.plot(t1, y1)
plt.plot(t1, y2)
plt.show()
