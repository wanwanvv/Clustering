import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn import metrics   # 评估模型
from sklearn.decomposition import PCA

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

#设定不同k值以运算
for k in range(2,10):
    clf = KMeans(n_clusters=k) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
    #start=time.clock()
    s = clf.fit(location) #加载数据集合
    #end=time.clock()
    centroids = clf.labels_
    # 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
    n_clusters_ = len(set(centroids)) - (1 if -1 in centroids else 0)
    #数据分析
    clusters_km = clf.labels_.tolist()
    result= {}
    for i in set(clusters_km):
        result[i]=clusters_km.count(i)
    # # 模型评估
    # print("#######################################################################################")
    print(clusters_km)
    # print("k=%d：" % k)
    print('估计的聚类个数为: %d' % n_clusters_)
    print("各类的统计个数为：")
    print(result)
    # print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, centroids))  # 每个群集只包含单个类的成员。
    # print("完整性: %0.3f" % metrics.completeness_score(labels_true, centroids))  # 给定类的所有成员都分配给同一个群集。
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, centroids))  # 同质性和完整性的调和平均
    # print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, centroids))
    # print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, centroids))
    # print("轮廓系数: %0.3f" % metrics.silhouette_score(location, centroids))
    # #print("time:")
    # #print(end-start)
    # print("#######################################################################################")
    #画出图来
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    #画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(length):
        #markIndex = int(clusterAssment[i, 0])
        plt.plot(location[i][0], location[i][1], mark[clf.labels_[i]]) #mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出质点，用特殊图型
    centroids =  clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
        #print centroids[i, 0], centroids[i, 1]
    plt.show()


# k=5
# clf = KMeans(n_clusters=k) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
# #start=time.clock()
# s = clf.fit(location) #加载数据集合
# #end=time.clock()
# centroids = clf.labels_
# # 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
# n_clusters_ = len(set(centroids)) - (1 if -1 in centroids else 0)
# #数据分析
# clusters_km = clf.labels_.tolist()
# np.savetxt("outPut.txt",clusters_km)
# result= {}
# for i in set(clusters_km):
#     result[i]=clusters_km.count(i)
#
# # 模型评估
# print("#######################################################################################")
# print(clusters_km)
# print("k=%d：" % k)
# print('估计的聚类个数为: %d' % n_clusters_)
# print("各类的统计个数为：")
# print(result)
# # print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, centroids))  # 每个群集只包含单个类的成员。
# # print("完整性: %0.3f" % metrics.completeness_score(labels_true, centroids))  # 给定类的所有成员都分配给同一个群集。
# # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, centroids))  # 同质性和完整性的调和平均
# # print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, centroids))
# # print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, centroids))
# # print("轮廓系数: %0.3f" % metrics.silhouette_score(location, centroids))
# # #print("time:")
# # #print(end-start)
# # print("#######################################################################################")

# R = list(range(256))
# random.shuffle(R)
# R = np.array(R)/255.0
# G = list(range(256))
# random.shuffle(G)
# G = np.array(G)/255.0
# B = list(range(256))
# random.shuffle(B)
# B = np.array(B)/255.0
# colors = []
# for i in range(256):
#     colors.append((R[i], G[i], B[i]))
#
# #画出图来
# mark = ['o', '^', '+', 's', 'd', '<', 'p']
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
# plt.show()