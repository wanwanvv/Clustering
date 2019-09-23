import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn import metrics   # 评估模型
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import cv2

#readFile
fileName = input("Enter the file's name: ")
location = []
labels_true = []
#读取txt文件中每一行的值
for line in open(fileName, "r"):
#每个输入数据以逗号隔开
    items = line.strip("\n").split(",")
    # labels_true.append(int(items.pop()))
    tmp = []
    for item in items:
        tmp.append(float(item))
    location.append(tmp)
location = np.array(location)
labels_true = np.array(labels_true)
length = len(location)

#####对地理坐标进行聚类
###球面距离计算
def distSLC(vecA, vecB):    #球面余弦定理
    a = math.sin(vecA[0,1]*math.pi/180) * math.sin(vecB[0,1]*math.pi/180)       #pi/180转换为弧度 ，pi  ,numpy
    b = math.cos(vecA[0,1]*math.pi/180) * math.cos(vecB[0,1]*math.pi/180) * math.cos(math.pi * (vecB[0,0]-vecA[0,0]) /180)
    return math.arccos(a + b)*6371.0

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
mark = ['o', '^', '+', 's', 'd', '<', 'p']
# #画出所有样例点 属于同一分类的绘制同样的颜色
# for i in range(length):
#     #markIndex = int(clusterAssment[i, 0])
#     plt.plot(location[i][0], location[i][1], color = colors[clf.labels_[i]], marker = '.') #mark[markIndex])
# mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
# # 画出质点，用特殊图型
# centroids =  clf.cluster_centers_
# for i in range(k):
#     plt.plot(centroids[i][0], centroids[i][1], mark[i % 10], markersize=6)
#     #print centroids[i, 0], centroids[i, 1]

k = 7
image_path = "E:/map.png"
clf = KMeans(n_clusters=k) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
s = clf.fit(location) #加载数据集合
centroids = clf.labels_
# 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
n_clusters_ = len(set(centroids)) - (1 if -1 in centroids else 0)
fig = plt.figure()  #可视化簇和簇质心。

####为了画出这幅图，首先创建一幅画，一个矩形
rect = [0.1, 0.1, 0.8, 0.8]   #创建矩形。
#使用唯一标记来标识每个簇。
axprops = dict(xticks=[], yticks=[])
ax0 = fig.add_axes(rect, label='ax0', **axprops)    #绘制一幅图，图0
# imgP = mpimg.imread(image_path)
# plt.savefig('map.jpg',bbox_inches='tight')
imgP = cv2.imread('map.jpg',cv2.IMREAD_COLOR)   #调用 imread 函数，基于一幅图像，来创建矩阵。
imgP = cv2.cvtColor(imgP, cv2.COLOR_BGR2RGB)
ax0.imshow(imgP)              #调用imshow ，绘制（基于图像创建）矩阵的图。
rect1 = [0.25, 0.26, 0.5, 0.45]#中国
plt.title('China terrorism events in last 20 year-GTD')
ax1 = fig.add_axes(rect1, label='ax1', frameon=False)   #绘制一幅新图，图1。 作用：使用两套坐标系统（不做任何偏移或缩放）。

for i in range(length):
    #markIndex = int(clusterAssment[i, 0])
    ax1.plot(location[i][0], location[i][1], color = colors[clf.labels_[i]], marker = '.') #mark[markIndex])
mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
# 画出质点，用特殊图型
centroids = clf.cluster_centers_
for i in range(k):
    ax1.plot(centroids[i][0], centroids[i][1], mark[i % 10], markersize=6)
plt.xlabel('longitude'), plt.ylabel('latitude')
plt.show()


