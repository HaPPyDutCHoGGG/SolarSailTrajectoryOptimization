import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры орбит планет (полуоси, эксцентриситеты)
semimajor_axes = [0.39, 0.72, 1.00, 1.52, 5.20, 9.58, 19.22, 30.05]  # в астрономических единицах (1 AU = 149.6 млн км)
eccentricities = [0.206, 0.007, 0.017, 0.093, 0.049, 0.056, 0.046, 0.010]

# Угловые скорости планет в радианах в секунду
angular_velocities = [2 * np.pi / 88, 2 * np.pi / 225, 2 * np.pi / 365, 2 * np.pi / 687, 2 * np.pi / 12, 2 * np.pi / 29.5, 2 * np.pi / 84, 2 * np.pi / 165]

# Генерация времени для одного оборота самой долгой орбиты (планеты Нептун)
time = np.linspace(0, 165 * 365, 10000)

# Инициализация координат и скоростей планет
planet_positions = [np.zeros((len(time), 3)) for _ in range(len(semimajor_axes))]
planet_velocities = [np.zeros((len(time), 3)) for _ in range(len(semimajor_axes))]

# Вычисление орбит планет
for i in range(len(semimajor_axes)):
    a = semimajor_axes[i]
    e = eccentricities[i]
    w = angular_velocities[i]
    
    r = a * (1 - e**2) / (1 + e * np.cos(w * time))
    
    planet_positions[i][:, 0] = r * np.cos(w * time)
    planet_positions[i][:, 1] = r * np.sin(w * time)
    
    v = np.sqrt(2 / r - 1 / a)
    planet_velocities[i][:, 0] = -v * np.sin(w * time)
    planet_velocities[i][:, 1] = v * np.cos(w * time)

# Цвета планет (примерные цвета)
planet_colors = ['gray', 'orange', 'royalblue', 'red', 'saddlebrown', 'goldenrod', 'lightseagreen', 'blue']

# Размеры планет (примерные радиусы)
planet_radii = [0.38, 0.95, 1, 0.53, 11.21, 9.45, 4.01, 3.88]

# Названия планет
planet_names = ['Меркурий', 'Венера', 'Земля', 'Марс', 'Юпитер', 'Сатурн', 'Уран', 'Нептун']

# Визуализация солнечной системы
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(semimajor_axes)):
    ax.plot(planet_positions[i][:, 0], planet_positions[i][:, 1], planet_positions[i][:, 2], color=planet_colors[i], label=planet_names[i], linewidth=1)
    ax.scatter([planet_positions[i][0, 0]], [planet_positions[i][0, 1]], [planet_positions[i][0, 2]], color=planet_colors[i], s=planet_radii[i] * 100)
    ax.text(planet_positions[i][0, 0], planet_positions[i][0, 1], planet_positions[i][0, 2], planet_names[i], fontsize=12)

ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Солнечная система')
ax.legend()

plt.show()