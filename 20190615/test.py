#coding=utf-8
import scipy.io as scio
from sklearn import svm
import numpy as np
import xlrd as xd
import matplotlib.pyplot as plt
#3.09数据处理以及时间数据初始化
data_path = "G:\dataset\data_1010309.mat"
data = scio.loadmat(data_path)
data_ar = data.get('data_train_1010309')
X = np.arange(1440)[:, None]
y = np.zeros(1440)
for i in X:
    y[i] = data_ar[i, 1]
Xt = np.linspace(0, 1439, 1440)[:, None]


def get_data(path):#从Excel读取数据
    da = xd.open_workbook(path)
    table = da.sheets()[0]
    y_get = table.col_values(1)
    y_get.remove('value')
    return y_get


#3.10数据处理
y_0310 = get_data('G:\\dataset\\20090310_101.xlsx')
#3.11数据处理
y_0311 = get_data('G:\\dataset\\20090311_101.xlsx')
#以0309数据为训练集
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y)
fig = plt.figure()
yt = clf.predict(Xt)


#作图函数
def display_result(time, value, postion, title_name, col):
    ax = fig.add_subplot(postion)
    ax.set(xlim=[0, 1440], ylim=[550, 700], title=title_name,
           ylabel='MTV', xlabel='time series')
    ax.plot(time, value, color=col, linewidth=0.7)


#以0309为训练集作图结果
display_result(Xt, y, 331, 'Section_1010309', 'blue')
display_result(Xt, yt, 331, 'Section_1010309', 'red')
display_result(Xt, y_0310, 332, 'Section_1010310', 'blue')
display_result(Xt, yt, 332, 'Section_1010310', 'red')
display_result(Xt, y_0311, 333, 'Section_1010311', 'blue')
display_result(Xt, yt, 333, 'Section_1010311', 'red')
#以0310为训练集
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0310)
yt = clf.predict(Xt)
display_result(Xt, y, 334, 'Section_1010309', 'blue')
display_result(Xt, yt, 334, 'Section_1010309', 'red')
display_result(Xt, y_0310, 335, 'Section_1010310', 'blue')
display_result(Xt, yt, 335, 'Section_1010310', 'red')
display_result(Xt, y_0311, 336, 'Section_1010311', 'blue')
display_result(Xt, yt, 336, 'Section_1010311', 'red')
#以0311为训练集
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0311)
yt = clf.predict(Xt)
display_result(Xt, y, 337, 'Section_1010309', 'blue')
display_result(Xt, yt, 337, 'Section_1010309', 'red')
display_result(Xt, y_0310, 338, 'Section_1010310', 'blue')
display_result(Xt, yt, 338, 'Section_1010310', 'red')
display_result(Xt, y_0311, 339, 'Section_1010311', 'blue')
display_result(Xt, yt, 339, 'Section_1010311', 'red')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()





