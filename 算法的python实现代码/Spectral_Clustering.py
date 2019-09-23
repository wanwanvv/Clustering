import numpy as np
from sklearn.cluster import SpectralClustering
import random
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib
from sklearn.metrics import calinski_harabaz_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics   # 评估模型

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
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

#Read data
fileName = input("Enter the file's name: ")
location = []
labels_true = []
#读取txt文件中每一行的值
for line in open(fileName, "r"):
#每个输入数据以逗号隔开
    items = line.strip("\n").split(",")
    labels_true.append(int(items.pop()))
    tmp = []
    for item in items:
        tmp.append(float(item))
    location.append(tmp)
location = np.array(location)
labels_true = np.array(labels_true)
length = len(location)

# 开始聚类
# 第一步利用ch指标来调参，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。
# 假设我们不知道聚类的数目，那么我们就要使用交叉验证了
n_cluster = [2,3,4,5,6]
gamma = [0.0001, 0.001, 0.01, 0.1,1,10]
score_list=[]

max_index = 0

for i in n_cluster:
    for j in gamma:
        model = SpectralClustering(n_clusters=i, gamma=j)
        model.fit(location)
        score = calinski_harabaz_score(location, model.labels_)
        tmps = []
        tmps.append(i)
        tmps.append(j)
        tmps.append(score)
        score_list.append(tmps)
        print("簇数："+str(i)+" sigmma:"+str(j)+" ch指数:"+str(score))
#找到score值最大的
score_list = np.array(score_list)
print(score_list)
max_index=score_list.argmax(axis=0)
print("ch指数最大时：簇数="+str(int(score_list[max_index[2]][0]))+" sigmma="+str(float(score_list[max_index[2]][1])))

model = SpectralClustering(n_clusters=int(score_list[max_index[2]][0]), gamma=float(score_list[max_index[2]][1]))
# model = SpectralClustering(n_clusters=31, gamma=0.01)
model.fit(location)
pre_y = model.labels_
# 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
n_clusters_ = len(set(pre_y)) - (1 if -1 in pre_y else 0)

#画图
plt.figure()
plt.title(u"聚类结果")
for i in range(length):
    plt.plot(location[i][0], location[i][1], color=colors[pre_y[i]], marker='.')
plt.xlabel('x'), plt.ylabel('y')
plt.show()

# 模型评估
print('估计的聚类个数为: %d' % n_clusters_)
print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, pre_y))  # 每个群集只包含单个类的成员。
print("完整性: %0.3f" % metrics.completeness_score(labels_true, pre_y))  # 给定类的所有成员都分配给同一个群集。
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, pre_y))  # 同质性和完整性的调和平均
print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, pre_y))
print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, pre_y))
print("轮廓系数: %0.3f" % metrics.silhouette_score(location, pre_y))