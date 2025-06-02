#ТИПЫ ДАННЫХ
СПИСОК ЛИСТ

Copy O(n)
Append[1] 0(1)
Pop last 0(1)
Pop intermediate [2] O(n)
Insert O(n)
Get Item O(1)
Set Item 0(1)
Delete Item O(n)
Iteration O(n)
Get Slice O(k)
Del Slice O(n)
Set Slice O(k+n)
Extend[1] O(k)
Sort O(n log n)
Multiply O(nk)
x in s O(n)
min(s), max(s) O(n)
Get Length O(1)




#СТЕК - ТАРЕЛОЧНИЦА
Добавить элемент сверху (push) – метод append
достать верхний элемент (pop) – метод pop
from collections import deque
class Stack:
        def __init__(self):
                self.items = deque()
        def is_empty(self):
                return len(self.items) == 0
        def push(self, item):
                self.items.append(item)
        def pop(self):
                if self.is_empty():
                        ...
        def peek(self):
                if self.is_empty():
                        ...
#ОЧЕРЕДИ КАК В СОВКЕ
добавить элемент справа(enqueue) – метод append
удалить элемент слева(dequeue) – метод popleft

Copy O(n)
append O(1)
appendleft O(1)
pop O(1)
popleft O(1)
extend O(k)
extendleft O(k)
rotate O(k)
remove O(n)

#КУЧА(РЕАЛИЗАЦИЯ ЕСТЬ НИЖЕ):
Модуль heapq реализует min-кучу, что экономит время на ее реализацию.
Примечательно, что модуль работает с любым имеющимся у вас списком.
Метод heapify превращает любой список в min-кучу:
import heapq
heap = [3, 6, 1, 5, 9, 2, 4, 6]
heapq.heapify(heap)

Для родителя с индексом i его потомки будут иметь индексы 2i+1 и 2i+2
Для потомка с индексом i родитель будет иметь индекс (i-1)//

heappush O(log n)
heappop O(log n)

#Словари
k in d O(1) O(n)
Copy[3] O(n) O(n)
Get Item O(1) O(n)
Set Item[1] O(1) O(n)
Delete Item O(1) O(n)
Iteration [3] O(n)

#МНОЖЕСТВО
x in s O(1) O(n)
операции остлаьные за O(n)








#МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ
#МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ
#МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ
#МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ
#МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ
#МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ  МАТЕМАТИЧЕСКИЕ АЛГОСЫ

# 1. НОД
делим а на б получаем остаток, если он = 0 то б нод
если не равен 0 то меняем а = б а б = остаток  и так далее пока остаток не будет равен 0
a = int(sys.argv[1])
b = int(sys.argv[2])

def nod(a,b):
	if(a < b):
		a, b = b, a
	while(b > 0):
		a, b = b, a % b
	return a
res = nod(a,b)
print(res)

# 2. Вычисление Пи
отношение точек попавшие в круг на все точки * 4 
приближенно равно пи. квадратсо стрроной 1 дан и вписаная в него окружность
import sys
import random

n = sys.argv[1]
square = 0
circle = 0 
for i in range(n):
	x = random.random()
	y = random.random()
	square += 1
	if x**2 + y**2 <= 1:
		circle += 1
res  = circle * 4 / square
print(res)


# 3. Нахождение нулей функции методом Ньютона(корень числа)
оценим точку т0, устанавливаем в точке где кас пересек ось х точку т1. аналогично т2 т3 и тд
вычисление корня числа с почти равен нахождению нулей функции х^2 - с.
в кач начальной оценки берем т0 = с, если т0 = т0/с, то т0 равен кв корню из с, если нет то делим на 2 и так далее
import sys
import math

eps = 1e-15
 c = float(sys.argv[1])
t = c
while abs(t - t/c) > eps:
	t = (t - t/c) / 2.0
print(t)

# 4. Представление числа в заданной системе счисления
a - число которое нужно представить б - система исч
сначала берем остаток от а на б, потом делим а на б и повторяем пока а больше 0
import sys

a = int(sys.argv[1])
b = int(sys.argv[2])

res = ' '
while(a>0):
	res = str(a%b) + res
	a = a//b
print(res)

# 5.Разложение числа на множители
берем число н и делим его на первый просто множитель пока это возможно
дальше след простое число оже делим пока возможно и так пока не получим 1
import sys
factor = 2
n = int(sys.argv[1])
res = ''

while(n>1):
	while(n % factor == 0):
		res += str(factor) + ' '
		n //= factor
	factor +=1
print(res)

# 6. Генерация булеана (множества всех подмножеств списка)
import sys
import copy

# n = int(sys.argv[1])
result = [[]]

for i in range(n):
    new = copy.deepcopy(result)  # копируем все текущие подмножества
    for subset in new:
        subset.append(i)  # добавляем текущий элемент i в каждое подмножество
    result += new  # объединяем с предыдущими подмножествами

print(result)

# 7.Нахождение списка простых чисел используя решето Эратосфена
import sys
n = int(sys.argv[1])
lst = list(range(2,n+1))
i = 0
while i < len(lst):
	prost = lst[i]
	mult = prost * prost
	if mult > n:
		break
	while mult <= n :
		if mult in lst:
			lst.remove(mult)
		mult += prost
	i += 1
print(lst)

# 8. Приведение матрицы к ступенчатому виду
import sys

mat_str = str(sys.argv[1])

def parse_matrix(m_str):
    mat = []
    rows = m_str.strip().split(',')
    for row in rows:
        mat.append([float(x) for x in row.strip().split()])
    return mat

def to_row_step(mat):
    rows = len(mat)
    cols = len(mat[0])

    for i in range(min(rows, cols)):
        if mat[i][i] == 0:
            for j in range(i + 1, rows):
                if mat[j][i] != 0:
                    mat[i], mat[j] = mat[j], mat[i]
                    break

        if mat[i][i] != 0:
            for j in range(i + 1, rows):
                factor = mat[j][i] / mat[i][i]
                for k in range(i, cols):
                    mat[j][k] -= factor * mat[i][k]

    return mat

matrix = parse_matrix(mat_str)
echelon_matrix = to_row_step(matrix)
print(echelon_matrix)


# 9. Вычисление интеграла функции (любой из методов)
import sys
import math
def func(x):
	if x == 0:
		return 1
	else:
		return math.sin(x)/x
a = float(sys.argv[1])
b = float(sys.argv[2])
n = int(sys.argv[3])
h = (b - a) / n
x = a
s1 = 0
s2 = 0
while(x < b):
	s1 += func(x) * h 
	s2 += func(x + h) * h 
	x += h 
print(s1)
print(s2)

# 10. Вычисление произведения матриц
import sys

def mat(matrices, input1):
    rows = input1.strip().split(',')
    matrix = []
    for row in rows:
        elements = row.strip().split()
        matrix.append([float(elem) for elem in elements])
    matrices.append(matrix)
    return matrices
maybi = sys.argv[1:]
if len(maybi)==0:
     print("Invalid input")
     exit(0)
matrices = []
for matrix_str in maybi:
    matrices = mat(matrices, matrix_str)

A = matrices[0]
matrices = matrices[1:]

for B in matrices:

    if len(A[0]) != len(B):
        print("Incompatible size of matrices")
        exit(0)

    C = [[0] * len(B[0]) for _ in range(len(A))]
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            plus = 0
            for e in range(len(A[0])):
                plus += A[i][e] * B[e][j]
            C[i][j] = plus

    A = C
print(A)

# 11. Вычисление степени числа при помощи рекурсии
import sys
a = int(sys.argv[1])
n = int(sys.argv[2])
def powernum(a,n):
	if n == 0:
		return 1
	elif n == 1:
		return a
	elif n % 2 == 0:
		return powernum(a * a, n // 2)
	else:
		return powernum(a * a, n // 2) * a
res = powernum(a,n)
print(res)


# 12. Генерация всех перестановок множества элементов
import sys

def lst(stroka):
    if len(stroka) == 1:
        return [stroka]

    spis = []
    for n in range(len(stroka)):
        flet = stroka[n]
        letters = stroka[:n] + stroka[n+1:]

        for i in lst(letters):
            spis.append(flet + i)
    return spis  

stroka = sys.argv[1]
res = sorted(lst(stroka))  
print(res)

#ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ
#ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ
#ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ
#ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ
#ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ ДРУГИЕ АЛГОРИТМЫ

#1.Бинарный поиск
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid  # Возвращаем индекс найденного элемента
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Элемент не найден


#2 Медленные сортировки: пузырьком, вставками, выбором
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):  # последний элемент уже на месте
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Сдвигаем элементы, которые больше key, вправо
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        # Ищем наименьший элемент в оставшейся части
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        # Меняем местами с текущим элементом
        arr[i], arr[min_index] = arr[min_index], arr[i]



#3 Сортировка слиянием
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # находим середину
        left_half = arr[:mid]
        right_half = arr[mid:]

        # рекурсивно сортируем обе половины
        merge_sort(left_half)
        merge_sort(right_half)

        # сливаем отсортированные половины
        i = j = k = 0

        # сравниваем элементы и сливаем
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # добавляем оставшиеся элементы
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


#4 Быстрая сортировка

def quick_sort(arr, low, high):
    if low < high:
        # разделение массива и получение индекса опорного элемента
        pi = partition(arr, low, high)

        # рекурсивно сортируем левую и правую части
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]  # опорный элемент
    i = low - 1        # индекс меньшего элемента

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # ставим опорный элемент на своё место
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


#5 Вычисление арифметического выражения (обратная польская запись)
def evaluaterpn(expression):
    stack = []
    tokens = expression.split()
    for token in tokens:
        if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            # Если токен — число (в том числе отрицательное), добавляем в стек
            stack.append(int(token))
        else:
            # Оператор: извлекаем два последних числа из стека
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                # Приведение к float для вещественного деления
                result = a / b
            else:
                raise ValueError(f"Неизвестный оператор: {token}")
            stack.append(result)
    # В стеке должно остаться одно значение — результат
    return stack[0]
# Пример использования
expr = "3 4 + 2 * 7 /"
result = evaluaterpn(expr)
print(f"Результат: {result}")



#6 Вычисление арифметического выражения (инфиксная запись)
def infix_to_rpn(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []
    tokens = tokenize(expression)
    for token in tokens:
        if token.isdigit() or is_number(token):
            output.append(token)
        elif token in precedence:
            while stack and stack[-1] != '(' and precedence.get(stack[-1], 0) >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # убираем '('
    while stack:
        output.append(stack.pop())
    return output
def evaluate_rpn(rpn_tokens):
    stack = []
    for token in rpn_tokens:
        if token.isdigit() or is_number(token):
            stack.append(float(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
    return stack[0]
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def tokenize(expression):
    import re
    # Разбивает строку на числа, операторы и скобки
    return re.findall(r'\d+\.\d+|\d+|[+*/()-]', expression)


# Пример использования
infix_expr = "3 + 4 * 2 / ( 1 - 5 )"
rpn = infix_to_rpn(infix_expr)
print("ОПЗ:", ' '.join(rpn))
result = evaluate_rpn(rpn)
print("Результат:", result)



#7 Реализация очереди с приоритетом при помощи бинарной кучи
class MinHeap:
    def __init__(self):
        self.items=[]
    def push(self,x):
        self.items.append(x)
        self._sift_up(len(self.items)-1)
    def pop(self):
        if not self.items:
            return None
        self.items[0], self.items[-1] = self.items[-1], self.items[0]
        min_item = self.items.pop()
        self._sift_down(0)
        return min_item
    def top(self):
        if len(self.items)==0:
            print('Heap is empty')
            return
        return self.items[0]
    def show(self):
        print(self.items)
    def _sift_up(self,i):
        while True:
            if i == 0:
                break
            parent_index=(i - 1) // 2
            if self.items[parent_index] > self.items[i]:
                self.items[parent_index], self.items[i] = self.items[i], self.items[parent_index]
                i = parent_index
            else:
                break
    def _sift_down(self, i):
        while True:
                if 2*i+2>=len(self.items):
                  break
                if self.items[2*i+1]<self.items[2*i+2]:child = 2*i+1
                else: child = 2*i+2

                if self.items[i] > self.items[child] :
                    self.items[child], self.items[i] = self.items[i], self.items[child]
                    i = child
                else: break
if __name__ == '__main__':
    heap = MinHeap()
    for i in [3, 6, 1, 5, 9, 2, 4, 6]:
        heap.push(i)
heap.pop()
heap.pop()
heap.show()


#8 Сортировка кучей
def heap_sort(arr):
    heap = MinHeap()
    for x in arr:
        heap.push(x)
    return [heap.pop() for _ in range(len(heap.items))]


#9 Обход графа в ширину
from collections import deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            queue.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)

    return result
# Пример графа
graph = {'A': ['B', 'C'],'B': ['A', 'D', 'E'],'C': ['A', 'F'],'D': ['B'],'E': ['B', 'F'],'F': ['C', 'E']}
# Запуск BFS и сохранение результата
visited_order = bfs(graph, 'A')
print(visited_order)

#10 Обход графа в глубину
def dfs(graph, start, visited=None, result=None):
    if visited is None:
        visited = set()
    if result is None:
        result = []
    visited.add(start)
    result.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, result)
    return result
# Пример графа
graph = {'A': ['B', 'C'],'B': ['D', 'E'],'C': ['F'],'D': [],'E': ['F'],'F': []}
# Запуск и сохранение результата
visited_vertices = dfs(graph, 'A')
print(visited_vertices)


#11. Нахождение кратчайших расстояний в графе методом Дейкстры
def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    visited = set()

    while len(visited) < len(graph):
        # Выбираем непосещённую вершину с минимальным расстоянием
        min_vertex = None
        min_distance = float('inf')
        for vertex in graph:
            if vertex not in visited and distances[vertex] < min_distance:
                min_distance = distances[vertex]
                min_vertex = vertex

        if min_vertex is None:
            break  # Все достижимые вершины посещены

        visited.add(min_vertex)

        for neighbor, weight in graph[min_vertex]:
            if neighbor not in visited:
                new_distance = distances[min_vertex] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance

    return distances

# Пример графа
graph = {
    'A': [('B', 5), ('C', 1)],
    'B': [('A', 5), ('C', 2), ('D', 1)],
    'C': [('A', 1), ('B', 2), ('D', 4), ('E', 8)],
    'D': [('B', 1), ('C', 4), ('E', 3), ('F', 6)],
    'E': [('C', 8), ('D', 3)],
    'F': [('D', 6)]
}

start_vertex = 'A'
distances = dijkstra(graph, start_vertex)
print(f"Кратчайшие расстояния от вершины {start_vertex}:")
for vertex, distance in distances.items():
    print(f"{vertex}: {distance}")


#12 Построение минимального остовного дерева алгоритмом Прима
def prim(graph, start):
    visited = set([start])
    mst = []
    while len(visited) < len(graph):
        min_edge = None
        min_weight = float('inf')
        # Ищем минимальное ребро (u, v), где u — посещён, v — ещё нет
        for u in visited:
            for v, weight in graph[u]:
                if v not in visited and weight < min_weight:
                    min_weight = weight
                    min_edge = (u, v, weight)
        if min_edge is None:
            # Граф может быть несвязным
            break
        u, v, weight = min_edge
        visited.add(v)
        mst.append(min_edge)
    return mst
# Пример графа (неориентированный)
graph = {'A': [('B', 4), ('H', 8)],'B': [('A', 4), ('H', 11), ('C', 8)],'C': [('B', 8), ('I', 2), ('F', 4), ('D', 7)],'D': [('C', 7), ('F', 14), ('E', 9)],'E': [('D', 9), ('F', 10)],'F': [('E', 10), ('D', 14), ('C', 4), ('G', 2)],'G': [('F', 2), ('I', 6), ('H', 1)],'H': [('A', 8), ('B', 11), ('G', 1), ('I', 7)],'I': [('C', 2), ('G', 6), ('H', 7)]}
start_vertex = 'A'
mst = prim(graph, start_vertex)
print("Ребра минимального остовного дерева (u, v, weight):")
for edge in mst:
    print(edge)

#13 Перебор с возвратом (бэктрекинг)
def backtrack(path, used, nums, results):
    # Если длина текущего пути равна длине исходного списка — нашли одну перестановку
    if len(path) == len(nums):
        results.append(path[:])
        return
    for i in range(len(nums)):
        if used[i]:
            continue
        # Выбираем элемент
        used[i] = True
        path.append(nums[i])
        # Погружаемся глубже
        backtrack(path, used, nums, results)
        # Возврат: отменяем выбор
        path.pop()
        used[i] = False
nums = [1, 2, 3]
results = []
backtrack([], [False] * len(nums), nums, results)
print(results)


#14 Размер монет (жадная стратегия и динамическое программирование
def greedy_coin_change(coins, amount):
    coins.sort(reverse=True)
    count = 0
    used_coins = []
    for coin in coins:
        if amount == 0:
            break
        use = amount // coin
        if use > 0:
            used_coins.extend([coin] * use)
            count += use
            amount -= use * coin
    if amount != 0:
        return -1, []  # нельзя разменять точно
    return count, used_coins
def dp_coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)  # чтобы восстановить решение
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin
    if dp[amount] == float('inf'):
        return -1, []
    # Восстановление списка монет
    res = []
    curr = amount
    while curr > 0:
        res.append(parent[curr])
        curr -= parent[curr]

    return dp[amount], res
# Тестовые кейсы
test_cases = [
    ([1, 5, 10, 25], 63),
    ([1, 3, 4], 6),
    ([2, 5], 3),
    ([1, 7, 10], 14),
    ([5, 10, 20], 0),
]
for coins, amount in test_cases:
    print(f"Монеты: {coins}, Сумма: {amount}")
    greedy_result = greedy_coin_change(coins[:], amount)  # копия списка, тк сортируем
    dp_result = dp_coin_change(coins, amount)
    print("  Жадный алгоритм:")
    if greedy_result[0] == -1:
        print("    Невозможно разменять")
    else:
        print(f"    Кол-во монет: {greedy_result[0]}, Монеты: {greedy_result[1]}")
    print("  Динамическое программирование:")
    if dp_result[0] == -1:
        print("    Невозможно разменять")
    else:
        print(f"    Кол-во монет: {dp_result[0]}, Монеты: {dp_result[1]}")
    print()

#15 Решение задачи о рюкзаке 0-1
def knapsack_01(weights, values, W):
    n = len(weights)
    # dp[i][w] - максимальная стоимость при рассмотрении первых i предметов и вместимости w
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Восстановление выбранных предметов
    res = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            res.append(i - 1)  # предмет i-1 взят
            w -= weights[i - 1]

    res.reverse()
    return dp[n][W], res


# Пример:
weights = [2, 3, 4, 5]
values = [3, 4, 5, 8]
W = 5

max_value, items = knapsack_01(weights, values, W)
print(f"Максимальная стоимость: {max_value}")
print(f"Взятые предметы (индексы): {items}")

# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# МИРИК МИРИК МИРИК МИРИК
# Шпаргалка для зачёта по структурам данных и алгоритмам

### I. Структуры данных

#### 1. Список (list)

lst = [1, 2, 3]
lst.append(4)
lst.insert(1, 10)
lst.remove(2)
x = lst.pop()

* Вставка/удаление в конец — O(1)
* Вставка/удаление в середину — O(n)

#### 2. Стек

stack = []
stack.append(1)
stack.append(2)
stack.pop()  # 2

* push/pop — O(1)

#### 3. Очередь

from collections import deque
queue = deque()
queue.append(1)
queue.popleft()

* enqueue/dequeue — O(1)

#### 4. Бинарная куча (min-heap)

import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappop(heap)  # 1

* Вставка/удаление — O(log n)

#### 5. Словарь (dict)

d = {'a': 1}
d['b'] = 2
x = d['a']

* Поиск/вставка — O(1) в среднем

#### 6. Множество (set)

s = {1, 2, 3}
s.add(4)
s.remove(2)

* Вставка/удаление/поиск — O(1) в среднем

#### 7. Граф (список смежности)

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A'],
    'D': ['B']
}


### II. Математические алгоритмы

#### 1. НОД (алгоритм Евклида)

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

#### 2. Вычисление числа Пи (метод Лейбница)

pi = 0
for k in range(100000):
    pi += (-1)**k / (2*k + 1)
pi *= 4

#### 3. Метод Ньютона

def newton(f, df, x0, eps=1e-6):
    while abs(f(x0)) > eps:
        x0 -= f(x0) / df(x0)
    return x0

#### 4. Система счисления

def to_base(n, base):
    digits = "0123456789ABCDEF"
    res = ""
    while n:
        res = digits[n % base] + res
        n //= base
    return res

#### 5. Разложение на множители

def factorize(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1
    if n > 1:
        factors.append(n)
    return factors

#### 6. Булеан

from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

#### 7. Решето Эратосфена

def sieve(n):
    is_prime = [True] * (n+1)
    is_prime[0:2] = [False, False]
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

#### 8. Ступенчатый вид (метод Гаусса) — пропущено для краткости

#### 9. Интеграл (трапеции)

def integral(f, a, b, n):
    h = (b - a) / n
    return h * (0.5*f(a) + sum(f(a + i*h) for i in range(1, n)) + 0.5*f(b))

#### 10. Произведение матриц

def matrix_mult(A, B):
    return [[sum(a*b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

#### 11. Рекурсивное возведение в степень

def power(x, n):
    return 1 if n == 0 else x * power(x, n - 1)

#### 12. Перестановки

from itertools import permutations
list(permutations([1, 2, 3]))

### III. Другие алгоритмы

#### 1. Бинарный поиск

def binary_search(arr, x):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] == x:
            return m
        elif arr[m] < x:
            l = m + 1
        else:
            r = m - 1
    return -1

#### 2. Сортировки:

* Пузырьком, вставками, выбором: O(n^2)
* Слиянием: O(n log n)
* Быстрая: ср. O(n log n), худш. O(n^2)

#### 3. Обратная польская запись (вычисление)

def eval_rpn(expr):
    stack = []
    for token in expr:
        if token in '+-*/':
            b = stack.pop()
            a = stack.pop()
            stack.append(eval(f'{a}{token}{b}'))
        else:
            stack.append(int(token))
    return stack[0]

#### 4. Очередь с приоритетом

#### 5. Обход в ширину (BFS)

from collections import deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if v not in visited:
            visited.add(v)
            queue.extend(graph[v])

#### 6. Обход в глубину (DFS)