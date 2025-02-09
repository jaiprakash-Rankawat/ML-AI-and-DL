import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='red', label='Sine Wave')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Sine Wave')
plt.plot(x,np.cos(x), color='blue', label='Cosine Wave')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Cosine Wave')
plt.show()


# bar Graph

people = ['Mark', 'John', 'Emily', 'Adam', 'Sarah']
heights = [178, 185, 163, 180, 172]
plt.bar(people, heights)
plt.xlabel('People')
plt.ylabel('Height')
plt.title('Bar Graph')
plt.show()


# Pie Chart

Languages = ['English', 'French', 'Spanish', 'German', 'Italian']
people = [85, 90, 75, 80, 70]

plt.pie(people, labels=Languages, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()


# Scatter Plot

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.scatter(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Scatter Plot')
plt.show()


# 3D Plot (Scatter Plot)

x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('3D Scatter Plot')
plt.show()