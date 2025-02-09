import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

# creating sin wave plot
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Sine Wave')
plt.show()

# creating cos wave plot
y = np.cos(x)
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Cosine Wave')
plt.show()

# creating perabola plot
x = np.linspace(-10, 10, 100)
y = x**2
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Parabola')
plt.show()

# creating hyperbola plot

y = [10,20,30,20,10]
x = [1,2,3,4,5]
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Hyperbola')
plt.show()

