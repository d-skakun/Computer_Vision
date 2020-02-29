import numpy as np

data = np.loadtxt("./input.txt", skiprows=2)
f = open("input.txt")
p = float(f.readline().strip())
n = int(f.readline().strip())
size_road = 0.5 # Половина принадлежит точкам дороги
threshold = n * size_road

def find_abcd(data):
    data_mean = data - np.mean(data, axis=0)
    # Матрица ковариации
    matrix = np.cov(data_mean.T) 
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    
    # Нормаль к плоскости - последний собственный вектор
    normal = eigenvectors[:, 2]
    point = np.mean(data, axis=0)

    a, b, c = normal
    d = -(np.dot(normal, point))
    
    return [np.round(i, 6) for i in [a, b, c, d]]

def optimaze(data, a, b, c, d):
    # Смотрим есть ли точки с разными знаками (разные классы)
    target = data.dot([a, b, c])
    more_idx, more_p = [], []
    less_idx, less_p = [], []
    for i, val in enumerate(target):
        # Если отклонение не больше чем p
        if val <= p:
            less_idx.append(i)
            less_p.append(val)
        else:
            more_idx.append(i)
            more_p.append(val)
  
    if len(more_p) > 0 and len(less_p) > 0:
        # Если есть разные знаки, то больший класс относим к дороге. Остальные выкидываем   
        if len(more_p) >= len(less_p):
            road_idx = more_idx
            road = more_p
        else:
            road_idx = less_idx
            road = less_p

        # Ищем параметры плоскости дороги еще раз
        a, b, c, d = find_abcd(data[road_idx, :])
        target_new = data[road_idx, :].dot([a, b, c])
        
        # Разделяем на классы дальше, если они находятся, 
        # так как плоскость усредняется по всем точкам, а препятствие может быть высоким
        is_need_optimize = np.unique(target_new > p, return_counts=True)
        if len(is_need_optimize[0]) > 1 and len(data[road_idx, :]) > threshold:
            return optimaze(data[road_idx, :], a, b, c, d)
        else:
            return a, b, c, d        
    else:
        # Если нет, то отдаем что было
        return a, b, c, d

coefs = find_abcd(data)
a, b, c, d = optimaze(data, *coefs)

print(a, b, c, d)