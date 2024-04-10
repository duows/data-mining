import matplotlib.pyplot as plt
import numpy as np

# Criando os dados
np.random.seed(0)
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)

# Criando o gráfico de dispersão 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando os dados
ax.scatter(x, y, z, c='r', marker='o')

# Rótulos dos eixos
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()