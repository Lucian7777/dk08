#coding=utf-8
import scipy.io as scio
from sklearn import svm
import numpy as np
import xlrd as xd
import matplotlib.pyplot as plt
import function as fun
# 归一化处理训练 以101区段预测102
CLASSNUM = 1440
SECTION = 101
data_ar = fun.get_data_by_section(101)
data_ar = [float(i) for i in data_ar]
Max = max(data_ar)
Min = min(data_ar)
for x in range(4320):
    data_ar[x] = (data_ar[x] - Min) / (Max - Min)
X = np.arange(CLASSNUM)[:, None]
Xt = np.linspace(0, CLASSNUM - 1, CLASSNUM)[:, None]
temp = np.zeros((1440, 1), dtype=np.float)
X = [int(i) for i in X]
y_0309 = fun.split_by_day(data_ar, X, 0)
y_0310 = fun.split_by_day(data_ar, X, CLASSNUM)
y_0311 = fun.split_by_day(data_ar, X, CLASSNUM * 2)
Y_0309 = fun.residual_data(CLASSNUM, y_0309)
Y_0310 = fun.residual_data(CLASSNUM, y_0310)
Y_0311 = fun.residual_data(CLASSNUM, y_0311)
x_ = np.vstack((Xt, Xt, Xt))
y_ = np.hstack((Y_0309, Y_0310, Y_0311))
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(x_, y_)
yt = clf.predict(Xt)
# 解归一化
data_ar = fun.get_data_by_section(102)
Max = max(data_ar)
Min = min(data_ar)
for x in range(1440):
    yt[x] = yt[x] * (Max - Min) + Min
y_0309_102 = fun.split_by_day(data_ar, X, 0)
y_0310_102 = fun.split_by_day(data_ar, X, CLASSNUM)
y_0311_102 = fun.split_by_day(data_ar, X, CLASSNUM * 2)
fig1 = plt.figure()
# e_101_0309 = (yt - y_0309_102) / y_0309_102
e_101_0310 = (yt - y_0310_102) / y_0310_102
e_101_0311 = (yt - y_0311_102) / y_0311_102
# fun.display_result(Xt, e_101_0309, fig, 331, 'Section_1020309', 'blue')
fun.display_result(Xt, e_101_0310, fig1, 221, 'Section_1020310', 'blue')
fun.display_result(Xt, e_101_0311, fig1, 222, 'Section_1020311', 'blue')
# 以102区段0309作为训练集做比较
clf_09 = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf_09.fit(Xt, y_0309_102)
yt_0309 = clf_09.predict(Xt)
e_090310 = (yt_0309 - y_0310_102) / y_0310_102
e_090311 = (yt_0309 - y_0311_102) / y_0311_102
fun.display_result(Xt, e_090310, fig1, 223, 'Section_1020310', 'blue')
fun.display_result(Xt, e_090311, fig1, 224, 'Section_1020311', 'blue')


# 102归一化训练
SECTION = 102
data_ar = fun.get_data_by_section(102)
data_ar = [float(i) for i in data_ar]
Max = max(data_ar)
Min = min(data_ar)
for x in range(4320):
    data_ar[x] = (data_ar[x] - Min) / (Max - Min)
X = np.arange(CLASSNUM)[:, None]
Xt = np.linspace(0, CLASSNUM - 1, CLASSNUM)[:, None]
temp = np.zeros((1440, 1), dtype=np.float)
X = [int(i) for i in X]
y_0309 = fun.split_by_day(data_ar, X, 0)
y_0310 = fun.split_by_day(data_ar, X, CLASSNUM)
y_0311 = fun.split_by_day(data_ar, X, CLASSNUM * 2)
clf_102_n = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf_102_n.fit(Xt, y_0309)
yt_n = clf_102_n.predict(Xt)
for x in range(1440):
    yt_n[x] = yt_n[x] * (Max - Min) + Min
e_102090310_n = (yt_n - y_0310_102) / y_0310_102
e_102090311_n = (yt_n - y_0311_102) / y_0311_102
clf_102 = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf_102.fit(Xt, y_0309_102)
yt = clf_102.predict(Xt)
e_102090310 = (yt - y_0310_102) / y_0310_102
e_102090311 = (yt - y_0311_102) / y_0311_102
fig2 = plt.figure()
fun.display_result(Xt, e_102090310_n, fig2, 221, 'Section_1020310', 'blue')
fun.display_result(Xt, e_102090311_n, fig2, 222, 'Section_1020311', 'blue')
fun.display_result(Xt, e_102090310, fig2, 223, 'Section_1020310', 'blue')
fun.display_result(Xt, e_102090311, fig2, 224, 'Section_1020311', 'blue')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()
