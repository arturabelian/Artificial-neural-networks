# Простейший перцептрон с сигмоидальной функцией активации.

import numpy as np

# Функция сигмойды.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Тренировочные данные.
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1) # Сид генератора.

synaptic_weights = 2 * np.random.random((3, 1)) -1 # Массив 3 на 1.

print("Случайная инициализация весов: ")
print(synaptic_weights)

# Обрастное распространение ошибки.

for i in range(20000):
    input_layer = training_inputs # Входной слой принимает на вход тренировочные данные.
    outputs = sigmoid(np.dot(input_layer, synaptic_weights)) # Выход определяется исходя из произведения входного слоя на слой весов, затем результат передается в функцию активации.
    err = training_outputs - outputs # Ошибка рассогласования между идеальным значением и реальным(получившимся при работе нейросети)
    adj = np.dot(input_layer.T, err * (outputs * (1 - outputs))) # Вычисление корректировки с учетом ошибки.
    synaptic_weights += adj # Корректировка слоя весов.

print("Вес после обучения: ")
print(synaptic_weights)
print("Результат после обучения: ")

print(outputs)

# Тест

new_inputs = np.array([1, 1, 0])

outputs = sigmoid(np.dot(new_inputs, synaptic_weights))

print("Новая ситуация: ")
print(outputs)


# ===========================
# 
# Случайная инициализация весов:
# [[-0.16595599]
#  [ 0.44064899]
#  [-0.99977125]]
# 
# Вес после обучения:
# [[ 10.38040701]
#  [ -0.20641179]
#  [ -4.98452047]]
# 
# Результат после обучения:
# [[ 0.00679672]
#  [ 0.99445583]
#  [ 0.99548516]
#  [ 0.00553614]]
# 
# Новая ситуация:
# [ 0.99996185]
# 
# 

