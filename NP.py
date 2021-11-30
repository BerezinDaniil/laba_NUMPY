'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 50))
line = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line

def animate(i):
    x=np.
    N = np.loadtxt("start.dat.txt")
    A = np.zeros((N.shape[0], N.shape[1]))
    np.fill_diagonal(A, 1)
    rows, cols = np.indices(A.shape)
    A[rows == cols + 1] = -1
    A[0][-1] = -1
    y =

    line.set_data(x, y)
    return line


anim = FuncAnimation(fig, animate, init_func=init,frames=200, interval=20, blit=True)

anim.save('sine_wave.gif', writer='imagemagick')


t=int(input()) #количество вагонов
A=[]
for i in range(t):
    A.append(input())#len(input())-количество мест в i-том купе
k=int(input())#количество купе в поезде
K=input()
m = int(input())
M=input()#iое число запрашивал iый gjkmpjdfntkm
train=''
for j in range(k):
    train+=A[int(K.split()[j])-1]
requests=''
for h in range(m):
    requests+=train[int(M.split()[h])-1]
print(requests)



s =input()
n=0
for i in range(0,len(s)+1):
    for j in range(i+1,len(s)+1):
        if len(list(set(s[i:j])))>=2:
            n+=1
print(n)


n = int(input())
N = input().split()
M = 100000000
N.sort()
for i in range(len(N) - 1):
    for j in range(len(N) - 1):
        if (j != i) and (j != (i - 1)) and (j != (i + 1)):
            Min = abs(abs(int(N[i]) - int(N[i + 1])) - abs(int(N[j]) - int(N[j + 1])))
            if Min < M:
                M = Min
print(M)





#Задача про лжецов и рыцарей
n = int(input())
A=[]
for i in range(n):
    k =int(input())
    if k>=n:
        k = n
    A.append(k)
B=[]
for i in range(min(A)+1,max(A),1):
    s=0
    for j in range(n):
        if A[j]>=i:
            s+=1
    if s==i:
        B.append(i)
if len(B)!=0:
    print(B[0])
else:
    print(-1)

import numpy as np
import matplotlib.pyplot as plt

array2 = np.zeros((1000000), dtype=np.longdouble)
for n in range(0, 1000000):
    array = np.ones((n,), dtype=np.longdouble)
    array1 = np.arange(1, n + 1)
    array2[n] = np.sum(np.divide(array, array1))
    print(n)
x = np.arange(0, 1000000)
y = array2

fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid()
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def creating_matrix():
    A = np.zeros((50, 50))
    np.fill_diagonal(A, 1)
    rows, cols = np.indices(A.shape)
    row_values = np.diag(rows, -1)
    col_values = np.diag(cols, -1)
    A[row_values, col_values] = -1
    A[0][-1] = -1
    return A


plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, 50), ylim=(0, 10))
ax.grid()
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.linspace(0, 50)
    N = np.loadtxt("start.txt", dtype=np.float64)
    if i > 0:
        if i == 1:
            n0 = np.loadtxt("start.txt", dtype=np.float64)
        else:
            n0 = np.loadtxt("start1.txt", dtype=np.float64)
        N = n0 - (0.5 * creating_matrix().dot(n0))
        # print(N)
        np.savetxt("start1.txt", N)
    Y = N
    line.set_data(x, Y)
    return line,


anim = FuncAnimation(fig, animate, init_func=init, frames=255, interval=0.5, blit=False)

anim.save('sine_wave.gif', writer='imagemagick')
