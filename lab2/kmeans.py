import numpy as np

import numpy as np

data = np.array([
    [8, 134, 2.25, 14, 8.5],
    [6, 141, 3.55, 14, 7.2],
    [6, 107, 1.75, 22, 7.8],
    [8, 128, 2.22, 13, 10.5],
    [9, 113, 1.45, 13, 13.1],
    [9, 141, 1.54, 22, 9.4],
    [8, 115, 2.15, 25, 11.3]
])

# data = np.array([
#     [30,    0.6,   2,   5],
#     [33,    0.6,   2.5, 5],
#     [50,    1,     1,   15],
#     [45,    0.8,   2,   10],
#     [20,    0.2,   1,   5],
#     [25,    0.6,   1,   20],
#     [203,   3.8,  10.5, 60]
# ])

# Параметри алгоритму
k = 2  # Кількість кластерів
max_iterations = 100  # Максимальна кількість ітерацій


# Функція для випадкової ініціалізації центроїдів
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


# Функція для обчислення відстані між двома точками
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Функція для оновлення кластерів та центроїдів
def update_clusters(data, centroids):
    clusters = [[] for _ in range(k)]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
    return clusters


# Функція для обчислення нових центроїдів
def update_centroids(clusters):
    return [np.mean(cluster, axis=0) for cluster in clusters]


# Основна функція для виконання кластеризації
def k_means(data, k, max_iterations):
    centroids = initialize_centroids(data, k)
    iteration = 0
    while iteration < max_iterations:
        clusters = update_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        iteration += 1
    return clusters, centroids


if __name__ == '__main__':
    clusters, centroids = k_means(data, k, max_iterations)
    print("Кластери:")
    for i, cluster in enumerate(clusters):
        print(f"Кластер {i + 1}:")
        for point in cluster:
            print(point)
        print()
    print("Центроїди кластерів:", centroids)
