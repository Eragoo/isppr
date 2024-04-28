import numpy as np


groups = [
    np.array([
        [8.1, 6.9, 6.2, 8.3],
        [9.4, 9.2, 8.5, 13.4],
        [14.1, 10.7, 12.8, 11.3],
        [14.1, 11.5, 2.25, 8.5],
        [3.55, 7.2, 1.75, 7.8],
        [2.22, 105, 1.45, 13.1],
        [1.85, 9.4, 2.15, 11.3]
    ]),
    np.array([
        [4.7, 6.1, 4.3, 5.8],
        [6.3, 5.0, 5.5, 13.8],
        [14.0, 11.8, 12.0, 14.3],
        [15.0, 12.9, 1.2, 1.7],
        [1.4, 1.9, 1.2, 1.4],
        [1.8, 10.72, 10.54, 12.83],
        [13.55, 14.67, 15.64, 14.78],
    ])
]

# groups = [
#     np.array([
#         [22.4, 17.1, 22],
#         [224.2, 17.1, 23],
#         [151.8, 14.9, 21.5],
#         [147.3, 13.6, 28.7],
#         [152.3, 10.5, 10.2]
#     ]),
#     np.array([
#         [46.8, 4.4, 11.1],
#         [29, 5.5, 6.1],
#         [52.1, 4.2, 11.8],
#         [37.1, 5.5, 11.9],
#         [64, 4.2, 12.9]
#     ])
# ]

# Обчислення середніх значень для кожної групи
means = [np.mean(group, axis=0) for group in groups]

# Обчислення кількості спостережень у кожній групі
n = [len(group) for group in groups]

# Обчислення матриць коваріації для кожної групи
covs = [np.cov(group, rowvar=False, bias=True) for group in groups]

# Обчислення об'єднаної матриці коваріації
cov_combined = np.average(covs, weights=n, axis=0)

# Обчислення об'єднаних векторів середніх
mean_combined = np.average(means, weights=n, axis=0)

# Обчислення дискримінантних коефіцієнтів
inv_cov = np.linalg.inv(cov_combined)
coefs = [np.dot(inv_cov, mean) for mean in means]


# Функція класифікації нових значень
def classify_new_value(x):
    discriminants = [np.dot(coef, x) - 0.5 * np.dot(mean, np.dot(inv_cov, mean)) for coef, mean in zip(coefs, means)]
    max_discriminant = max(discriminants)
    return chr(ord('a') + discriminants.index(max_discriminant))


if __name__ == '__main__':
    # Приклад класифікації нового значення
    print("Класифікація [7.5, 10.23, 2.31, 10.45]:", classify_new_value([7.5, 10.23, 2.31, 10.45]))
    print("Класифікація [6.5, 12.02, 1.36, 14.41]:", classify_new_value([6.5, 12.02, 1.36, 14.41]))
