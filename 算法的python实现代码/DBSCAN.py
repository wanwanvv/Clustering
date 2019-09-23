import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图
from sklearn.decomposition import PCA

# StandardScaler作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。

#用的是Aggregation.txt，因为其有标签
#readFile
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

location = StandardScaler().fit_transform(location)

#参数设置:Aggregation.txt()
#较高的 min_samples 或者较低的 eps表示形成聚类所需的较高密度。
# 调用密度聚类  DBSCAN(0.3,20)

db = DBSCAN(eps=0.2, min_samples=10).fit(location)
# print(db.labels_)  # db.labels_为所有样本的聚类索引，没有聚类索引为-1
# print(db.core_sample_indices_) # 所有核心样本的索引
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# 设置一个样本个数长度的全false向量
core_samples_mask[db.core_sample_indices_] = True #将核心样本部分设置为true
labels = db.labels_

# 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# # 模型评估
print('估计的聚类个数为: %d' % n_clusters_)
print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, labels))  # 每个群集只包含单个类的成员。
print("完整性: %0.3f" % metrics.completeness_score(labels_true, labels))  # 给定类的所有成员都分配给同一个群集。
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))  # 同质性和完整性的调和平均
print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("轮廓系数: %0.3f" % metrics.silhouette_score(location, labels))

# Plot result
mark1 = []
merk2 = []
mark1 = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
mark2 = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:  # 聚类结果为-1的样本为离散点
        # 使用黑色绘制离散点
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)  # 将所有属于该聚类的样本位置置为true

    xy = location[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor=tuple(col), markersize=6)

    xy = location[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor=tuple(col), markersize=3)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()