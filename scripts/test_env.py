import numpy as np
import matplotlib.pyplot as plt

print("Hello Ship MARL Project!")
print("NumPy version:", np.__version__)

# Create a simple trajectory
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Test Trajectory")
plt.title("First Test Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()