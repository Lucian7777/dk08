# coding=utf-8
from sklearn import svm
import numpy as np
import xlrd as xd
import scipy.io as scio
import matplotlib.pyplot as plt
import function as fun
CLASSSNUM = 1440
# 对117区段需要另做处理，如下注释
# data_path = "D:\matlab2016\\bin\\117.mat"
# data = scio.loadmat(data_path)
# data_ar = data.get('data')
# dataSize = data_ar.shape[0]/3
# print(dataSize)
# X = np.arange(dataSize)[:, None]
# y_0309 = np.zeros(dataSize)
# for i in X:
#     y_0309[i] = data_ar[i]
# Xt = np.linspace(0, dataSize-1, dataSize)[:, None]
# y_0310 = np.zeros(dataSize)
# for i in X:
#     y_0310[i] = data_ar[i+dataSize]
# y_0311 = np.zeros(dataSize)
# for i in X:
#     y_0311[i] = data_ar[i+dataSize*2]
# print(len(y_0311))
# 选取区段
SECTION = 101
data_ar = fun.get_data_by_section(SECTION)
X = np.arange(CLASSSNUM)[:, None]
Xt = np.linspace(0, CLASSSNUM - 1, CLASSSNUM)[:, None]
y_0309 = fun.split_by_day(data_ar, X, 0)
y_0310 = fun.split_by_day(data_ar, X, CLASSSNUM)
y_0311 = fun.split_by_day(data_ar, X, CLASSSNUM * 2)
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0309)
yt = clf.predict(Xt)
e_090309 = (yt - y_0309) / y_0309
e_090310 = (yt - y_0310) / y_0310
e_090311 = (yt - y_0311) / y_0311
fig = plt.figure()
# fun.display_result(Xt, e_090309, fig, 321, 'Section_' + str(SECTION) + '0309', 'blue')
fun.display_result(Xt, e_090310, fig, 321, 'Section_' + str(SECTION) + '0310', 'blue')
fun.display_result(Xt, e_090311, fig, 322, 'Section_' + str(SECTION) + '0311', 'blue')

clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0310)
yt = clf.predict(Xt)
e_100309 = (yt - y_0309) / y_0309
e_100310 = (yt - y_0310) / y_0310
e_100311 = (yt - y_0311) / y_0311
fun.display_result(Xt, e_100309, fig, 323, 'Section_' + str(SECTION) + '0309', 'blue')
# fun.display_result(Xt, e_100310, fig, 335, 'Section_' + str(SECTION) + '0310', 'blue')
fun.display_result(Xt, e_100311, fig, 324, 'Section_' + str(SECTION) + '0311', 'blue')
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0311)
yt = clf.predict(Xt)
e_110309 = (yt - y_0309) / y_0309
e_110310 = (yt - y_0310) / y_0310
e_110311 = (yt - y_0311) / y_0311
fun.display_result(Xt, e_110309, fig, 325, 'Section_' + str(SECTION) + '0309', 'blue')
fun.display_result(Xt, e_110310, fig, 326, 'Section_' + str(SECTION) + '0310', 'blue')
# fun.display_result(Xt, e_110311, fig, 339, 'Section_' + str(SECTION) + '0311', 'blue')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()

