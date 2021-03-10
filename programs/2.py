# Нейросеть.

# Инициализация — задание количества входных, скрытых и выходных узлов.
# Тренировка — уточнение весовых коэффициентов в процессе обработки предоставленных для обучения сети тренировочных примеров.
# Опрос — получение значений сигналов с выходных узлов после предоставления значений входящих сигналов.

import numpy
# scipy.special для функции сигмойды expit()
import scipy.special

# Определение класса нейросети.
class Neuralnetwork:

############################## ИНИЦИАЛИЗАЦИЯ ##############################

    # Инициализация нейросети.
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Задаем кол-во узлов в слоях.
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Весовые матрицы связей, wih и who.
        
        # wih - Матрица весов (weight input-hidden) между входным слоем и скрытым.
        # who - Матрица весов (weight hidden-output)  между скрытым слоем и выходным.
         
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # Рейт обучения.
        self.lr = learningrate
        
        # Сигмойда - функция активации используется через анонимную лямбда функцию.
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

############################## ТРЕНИРОВКА ##############################
# Тренировка включает две фазы: первая — это расчет выходного сигнала, что и делает функция query(),
# а вторая — обратное распространение ошибок, информирующее о том, каковы должны быть поправки к весовым коэффициентам.

# Первая часть — расчет выходных сигналов для заданного тренировочного примера. Это ничем не отличается от того, что мы
# уже можем делать с помощью функции query().

# Вторая часть — сравнение рассчитанных выходных сигналов с желаемым ответом и обновление весовых коэффициентов связей
# между узлами на основе найденных различий.

# Единственным отличием является введение дополнительного параметра targets_list, передаваемого при вызове функции,
# поскольку невозможно тренировать сеть без предоставления ей тренировочных примеров, которые включают желаемые или целевые значения.

# Основной задачи тренировки сети — уточнению весов на основе расхождения между фактическими и целевыми значениями.

    # Тренировка нейросети.
    def train(self, inputs_list, targets_list):

        # Конвертирование списка входных сигналов в двумерную транспонированную матрицу.
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Вычисление входящих сигналов для скрытого слоя.
        hidden_inputs = numpy.dot(self.wih, inputs)

        # Вычисление исходящих сигналов для скрытого слоя.
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Вычисление входящих сигналов для выходного слоя.
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Вычисление исходящих сигналов для выходного слоя.
        final_outputs = self.activation_function(final_inputs)

# Для весов связей между скрытым и выходным слоями используется output_errors.
     
        # Ошибка выходного слоя это (целевое значение - фактическое значение).
        output_errors = targets - final_outputs

# Далее мы должны рассчитать обратное распространение ошибок для узлов скрытого слоя.
# Для весов связей между входным и скрытым слоями hidden_errors.

        # Ошибка скрытого слоя hidden_errors это ошибки выходного слоя output_errors, распределенные пропорционально
        # весовым коэффициентам связи и рекомбинированные на скрытых узлах.

        hidden_errors = numpy.dot(self.who.T, output_errors) # Error_hide = W_ho.T * Error_o

# Обновление веса связи между двумя узлами.

# Величина а — это коэффициент обучения, а сигмоида — это функция активации. Последний член выражения — это
# транспонированная (т) матрица исходящих сигналов предыдущего слоя. В данном случае транспонирование означает
# преобразование столбца выходных сигналов в строку.
        
        # Обновление весов для связи между скрытым и выходным слоем.
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # Обновление весов для связи между входным и скрытым слоем.
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

# numpy.dot - Матричное умножение.

        pass

############################## ОПРОС ##############################
# Функция query() принимает в качестве аргумента входные данные нейронной сети и возвращает ее выходные данные.

# При этом, по мере распространения сигналов мы должны сглаживать их, используя весовые коэффициенты связей между
# соответствующими узлами, суммирование, а также применять сигмоиду для уменьшения выходных сигналов узлов.

# Получение входящих сигналов для узлов скрытого слоя путем сочетания матрицы весовых коэффициентов связей между
# входным и скрытым слоями с матрицей входных сигналов.

# X_скрытый = W_входной_скрытый * I_входные_сигналы

    # Опрос нейросети.
    def query(self, inputs_list):

        # Конвертирование входных сигналов в двумерную транспонированную матрицу.
        inputs = numpy.array(inputs_list, ndmin=2).T # 
        
        # Вычисление сингалов в скрытом слое.
        hidden_inputs = numpy.dot(self.wih, inputs) # HI = I * WIH

        # Вычисление сигналов выходящих из скрытого слоя.
        hidden_outputs = self.activation_function(hidden_inputs) # HO = SIG(HI)
        
        # Вычисление сигналов в конечно выходном слое.
        final_inputs = numpy.dot(self.who, hidden_outputs) # FI = HO * WHO

        # Вычисление сигналов выходящих из выходного слоя.
        final_outputs = self.activation_function(final_inputs) # FO = SIG(FI)
        
        return final_outputs


# Вместо того чтобы жестко задавать их в коде, мы предусмотрим установку соответствующих значений в виде
# параметров во время создания объекта нейронной сети. Благодаря этому можно будет без труда создавать новые
# нейронные сети различного размера.

# Мы хотим, чтобы один и тот же класс мог создавать как небольшие нейронные сети, так и очень большие, требуя
# лишь задания желаемых размеров сети в качестве параметров.

# Число входных, скрытых и выходных узлов.
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# Кроме того, нам нельзя забывать о рейте обучения.

# Рейт обучения 0.3
learning_rate = 0.3

# Создание экземпляра нейронной сети.
n = Neuralnetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# Тест query (Просто так)
n.query([1.0, 0.5, -1.5])

# array([[ 0.43461026],
#        [ 0.40331273],
#        [ 0.56675401]])
