#coding=utf-8
import scipy.io as scio
from sklearn import svm
import numpy as np
import xlrd as xd
import matplotlib.pyplot as plt
#3.09数据处理以及时间数据初始化
data_path = "D:\matlab2016\\bin\\103.mat"
data = scio.loadmat(data_path)
data_ar = data.get('data')
dataSize = data_ar.shape[0]/3
print(dataSize)
X = np.arange(dataSize)[:, None]
y = np.zeros(dataSize)
for i in X:
    y[i] = data_ar[i]
Xt = np.linspace(0, dataSize-1, dataSize)[:, None]
y_0310 = np.zeros(dataSize)
for i in X:
    y_0310[i] = data_ar[i+dataSize]
y_0311 = np.zeros(dataSize)
for i in X:
    y_0311[i] = data_ar[i+dataSize*2]
print(len(y_0311))
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y)
fig = plt.figure()
yt = clf.predict(Xt)


#作图函数
def display_result(time, value, postion, title_name, col):
    ax = fig.add_subplot(postion)
    ax.set(xlim=[0, 1440], ylim=[value.min()-50, value.max()+50], title=title_name,
           ylabel='MTV', xlabel='time series')
    ax.plot(time, value, color=col, linewidth=0.7)


#以0309为训练集作图结果
display_result(Xt, y, 331, 'Section_1020309', 'blue')
display_result(Xt, yt, 331, 'Section_1020309', 'red')
display_result(Xt, y_0310, 332, 'Section_1020310', 'blue')
display_result(Xt, yt, 332, 'Section_1020310', 'red')
display_result(Xt, y_0311, 333, 'Section_1020311', 'blue')
display_result(Xt, yt, 333, 'Section_1020311', 'red')
#以0310为训练集
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0310)
yt = clf.predict(Xt)
display_result(Xt, y, 334, 'Section_1020309', 'blue')
display_result(Xt, yt, 334, 'Section_1020309', 'red')
display_result(Xt, y_0310, 335, 'Section_1020310', 'blue')
display_result(Xt, yt, 335, 'Section_1020310', 'red')
display_result(Xt, y_0311, 336, 'Section_1020311', 'blue')
display_result(Xt, yt, 336,  'Section_1020311',  'red')
#以0311为训练集
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0311)
yt = clf.predict(Xt)
display_result(Xt, y, 337, 'Section_1020309', 'blue')
display_result(Xt, yt, 337, 'Section_1020309', 'red')
display_result(Xt, y_0310, 338, 'Section_1020310', 'blue')
display_result(Xt, yt, 338, 'Section_1020310', 'red')
display_result(Xt, y_0311, 339, 'Section_1020311', 'blue')
display_result(Xt, yt, 339, 'Section_1020311', 'red')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()






