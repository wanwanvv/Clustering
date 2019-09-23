#在Anaconda的IDE下安装的python3.7版本
#numpy库-数组处理，matplotlib库-画图函数
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
from sklearn.preprocessing import StandardScaler
from sklearn import metrics   # 评估模型
from sklearn.decomposition import PCA

MAX = 1000000

def nearestNeighbor(index):
    dd = MAX
    neighbor = -1
    for i in range(length):
        if dist[index, i] < dd and rho[index] < rho[i]:
            dd = dist[index, i]
            neighbor = i
    if result[neighbor] == -1:
        result[neighbor] = nearestNeighbor(neighbor)
    return result[neighbor]

#Read data
percent=float(input("Enter the percent: "))
fileName = input("Enter the file's name: ")
location = []
label = []
#读取txt文件中每一行的值
for line in open(fileName, "r"):
#每个输入数据以逗号隔开
    items = line.strip("\n").split(",")
    # label.append(int(items.pop()))
    tmp = []
    for item in items:
        tmp.append(float(item))
    location.append(tmp)
location = np.array(location)
label = np.array(label)
length = len(location)

#Caculate distance
dist = np.zeros((length, length))
ll = []
begin = 0
while begin < length-1:
    end = begin + 1
    while end < length:
        dd = np.linalg.norm(location[begin]-location[end])
        dist[begin][end] = dd
        dist[end][begin] = dd
        ll.append(dd)
        end = end + 1
    begin = begin + 1
ll = np.array(ll)
# Algorithm        
#percent = float(input("Enter the average percentage of neighbours: "))
#percent = 2.0
position = int(len(ll) * percent / 100)
sortedll = np.sort(ll)
dc = sortedll[position] #阈值
#求点的局部密度(local density)
rho = np.zeros((length, 1))
begin = 0
while begin < length-1:
    end = begin + 1
    while end < length:
        rho[begin] = rho[begin] + np.exp(-(dist[begin][end]/dc) ** 2)
        rho[end] = rho[end] + np.exp(-(dist[begin][end]/dc) ** 2)
        #if dist[begin][end] < dc:
        #    rho[begin] = rho[begin] + 1
        #    rho[end] = rho[end] + 1
        end = end + 1
    begin = begin + 1

#求比点的局部密度大的点到该点的最小距离
delta = np.ones((length, 1)) * MAX
maxDensity = np.max(rho)
begin = 0
while begin < length:
    if rho[begin] < maxDensity:
        end = 0
        while end < length:
            if rho[end] > rho[begin] and dist[begin][end] < delta[begin]:
                delta[begin] = dist[begin][end]
            end = end + 1
    else:
        delta[begin] = 0.0
        end = 0
        while end < length:
            if dist[begin][end] > delta[begin]:
                delta[begin] = dist[begin][end]
            end = end + 1
    begin = begin + 1

rate1 = 0.2
halo1 =0.2
#Aggregation Spiral 0.6
#Jain Flame 0.8
#D31 0.75
#R15 0.6
#Compound 0.5
#Pathbased 0.2
#panelC.txt 0.2
#panelB.txt 0.2 噪点0.2
#MopsiLocations2012-Joensuu.txt 0.07
thRho = rate1 * (np.max(rho) - np.min(rho)) + np.min(rho)
thRho_halo = halo1 * (np.max(rho) - np.min(rho)) + np.min(rho)

rate2 = 0.2
halo2 = 0.03
#Aggregation Spiral 0.2
#Jain Flame 0.2
#D31 0.05
#R15 0.1
#Compound 0.08
#Pathbased 0.4
#panelC.txt 0.2
#panelB.txt 0.25 噪点 0.003
#MopsiLocations2012-Joensuu.txt 0.2
thDel = rate2 * (np.max(delta) - np.min(delta)) + np.min(delta)
thDel_halo = halo2 * (np.max(delta) - np.min(delta)) + np.min(delta)

#确定聚类中心
result = np.ones(length, dtype=np.int) * (-1)
halo = np.ones(length, dtype=np.int) * (-1)
center = 0
#items = range(length)
#random.shuffle(items)
for i in range(length): #items:
    if rho[i] > thRho and delta[i] > thDel:
        result[i] = center
        center = center + 1
#赋予每个点聚类类标
for i in range(length):
    dist[i][i] = MAX

for i in range(length):
    if result[i] == -1:
        result[i] = nearestNeighbor(i)
    else:
        continue

#确定噪声点
for i in range(length):
    if rho[i] < thRho_halo and delta[i] > thDel_halo:
        halo[i] = 1;

plt.plot(rho, delta, '.')
plt.xlabel('rho'), plt.ylabel('delta')
plt.show()

R = list(range(256))
random.shuffle(R)
R = np.array(R)/255.0
G = list(range(256))
random.shuffle(G)
G = np.array(G)/255.0
B = list(range(256))
random.shuffle(B)
B = np.array(B)/255.0
colors = []
for i in range(256):
    colors.append((R[i], G[i], B[i]))

plt.figure()
for i in range(length):
    if halo[i] == 1:
        plt.plot(location[i][0], location[i][1], color=(0, 0, 0), marker='+')
    else:
        index = result[i]
        plt.plot(location[i][0], location[i][1], color=colors[index], marker='.')
plt.xlabel('x'), plt.ylabel('y')
plt.show()


# plt.figure()
# for i in range(length):
#     index = label[i]
#     plt.plot(location[i][0], location[i][1], color = colors[index], marker = '.')
# plt.xlabel('x'), plt.ylabel('y')
# plt.show()


print("聚类的标签值为VS原来的标签值为：")
for j in range(length):
    print(str(result[j])+"   ",end=" "),
print('')
# for k in range(length):
#     print(str(label[k])+"   ",end="   "),

#统计各分类的个数
result_list=[]
result_list=np.unique(result)
result_num=len(result_list)
result_labs=[0]*result_num
print("聚类后的类别数为：%d" % result_num)
for i in range(result_num):
    for j in range(length):
        if result[j] == result_list[i]:
            result_labs[i] += 1
for k in range(result_num):
    print("tag="+str(k)+"的个数为: "+str(result_labs[k]))
