import numpy as np
import matplotlib.pyplot as plt
import random
import math
MAX=1000000
def nearestNeighbor(index):
    dd=MAX
    neighbor=-1
    for i in list(range(length)):
        if dist[index,i]<dd and rho[index] < rho[i]:
            dd=dist[index,i]
            neighbor=i
    if result[neighbor]==-1:
        result[neighbor]=nearestNeighbor(neighbor)
    return result[neighbor]

filename=input("Enter the file's name:")
location=[]
label=[]
for line in open(filename,"r"):
    items=line.strip("\n").split(",")
    label.append(int(items.pop()))#pop删除最后一个元素，并返回该元素的值
    tmp=[]
    for item in items:
        tmp.append(item)
    location.append(tmp)
location=np.array(location)
label=np.array(label)
length=len(location)

dist=np.zeros((length,length))
ll=[]
begin=0
while begin <length-1:
    end=begin+1
    while end<length:
        dd=np.linalg.norm(location[begin]-location[end])
        dist[begin][end]=dd
        dist[end][begin]=dd
        ll.append(dd)
        end=end+1
    begin=begin+1
ll=np.array(ll)

percent=2.0
position=int(len(ll)*percent/100)
sortedll=np.sort(ll)
dc=sortedll[position]

rho=np.zeros((length,1))
begin=0
while begin<length-1:
    end=begin+1
    while end<length:
        rho[begin]=rho[begin]+math.exp(-(dist[begin][end]//dc**2))
        rho[end] = rho[end] + math.exp(-(dist[begin][end] / dc) ** 2)
        end=end+1
    begin=begin+1

delta=np.ones((length,1))*MAX
maxDensity=np.max(rho)
begin=0
while begin <length:
    if rho[begin]<maxDensity:
        end=0
        while end <length:
            if rho[end]>rho[begin] and dist[begin][end]<delta[begin]:
                delta[begin]=dist[begin][end]
            end=end+1
    else:
        delta[begin]=0.0
        end=0
        while end <length:
            if dist[begin][end]>delta[begin]:
                delta[begin]=dist[begin][end]
            end=end+1
    begin=begin+1
rate1=0.1
thRho=rate1*(np.max(rho)-np.min(rho))+np.min(rho)
rate2=0.016
thDel=rate2*(np.max(delta)-np.min(delta))+np.min(delta)

result=np.ones(10,dtype=np.int)*(-1)
center=0
for i in list(range(length)):
    if rho[i]>thRho and delta[i]>thDel:
        result[i]=center
        center=center+1
for i in list(range(length)):
    dist[i][i]=MAX
for i in list(range(length)):
    if result[i]==-1:
        result[i]=nearestNeighbor(i)
    else:
        continue

plt.plot(rho,delta,'.')
plt.xlabel('rhp'),plt.ylabel('delta')
plt.show()

R=list(range(256))
random.shuffle(R)
R=np.array(R)/255.0
