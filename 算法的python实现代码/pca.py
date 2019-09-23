import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图
from sklearn.decomposition import PCA

#readFile
fileName = input("Enter the file's name: ")
location = []
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
length = len(location)

pca=PCA(n_components=2)
newData=pca.fit_transform(location)
print(newData)