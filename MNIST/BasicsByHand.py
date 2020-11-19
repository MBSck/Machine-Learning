import numpy as np
import matplotlib.pyplot as plt

# Doing calculation of the analytic differentiation by hand


def f(x):
    return x*np.exp(-x**2)+1/4


def df(x):
    return np.exp(-x**2)-2*(x**2)*np.exp(-x**2)


def gradient_minimum(f, df, x0, step, n_iterations=10):
    """Going toward the minimum due to first derivative and - in front.
    Depends crucially on the step size"""
    for i in range(n_iterations):
        x0 = x0 - df(x0)*step
    return x0


print(gradient_minimum(f, df, 0.1, 0.01, 100))
print(f(gradient_minimum(f, df, 0.1, 0.01, 100)))

x = np.arange(-3, 3, 0.01)

plt.plot(x, f(x))
plt.plot(x, f(0.1)+df(0.1)*(x-0.1))
plt.scatter([0.1], [f(0.1)], color="green")
plt.axhline(y=0, color="black")
plt.ylim(-0.5, 1)
plt.show()

