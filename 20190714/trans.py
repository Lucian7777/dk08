#coding=utf-8
#两组数据作为训练集
from sklearn import svm
import numpy as np
import xlrd as xd
import matplotlib.pyplot as plt
import function as fun
CLASSSNUM = 1440
SECTION = 101
data_ar = fun.get_data_by_section(SECTION)
X = np.arange(CLASSSNUM)[:, None]
Xt = np.linspace(0, CLASSSNUM - 1, CLASSSNUM)[:, None]
y_0309 = fun.split_by_day(data_ar, X, 0)
y_0310 = fun.split_by_day(data_ar, X, CLASSSNUM)
y_0311 = fun.split_by_day(data_ar, X, CLASSSNUM * 2)
#以9日数据训练
clf_09 = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf_09.fit(Xt, y_0309)
yt_0309 = clf_09.predict(Xt)
e_090311 = (yt_0309 - y_0311) / y_0311
fig = plt.figure()
fun.display_result(Xt, e_090311, fig, 131, 'Section_' + str(SECTION) + '0311', 'blue')
#以10日数据训练
clf_10 = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf_10.fit(Xt, y_0310)
yt_0310 = clf_10.predict(Xt)
e_100311 = (yt_0310 - y_0311) / y_0311
fun.display_result(Xt, e_100311, fig, 132, 'Section_' + str(SECTION) + '0311', 'blue')
#以9日和10日数据训练
x_0910 = np.vstack((Xt, Xt))
y_0910 = np.hstack((y_0310, y_0309))
clf_0910 = svm.SVR(C=6, cache_size=200, coef0=0.0, degree=2, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf_0910.fit(x_0910, y_0910)
yt_0910 = clf_0910.predict(Xt)
e_0910311 = (yt_0910 - y_0311) / y_0311
fun.display_result(Xt, e_0910311, fig, 133, 'Section_' + str(SECTION) + '0311', 'blue')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()

