#coding=utf-8
from sklearn import svm
import numpy as np
import xlrd as xd
import matplotlib.pyplot as plt
import function as fun
TIME = 1440
data_ar = fun.get_data_by_section(104)
X = np.arange(TIME)[:, None]
y_0309 = fun.split_by_day(data_ar, X, 0)
Xt = np.linspace(0, TIME - 1, TIME)[:, None]
y_0310 = fun.split_by_day(data_ar, X, TIME)
y_0311 = fun.split_by_day(data_ar, X, TIME * 2)
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0309)
yt = clf.predict(Xt)
residual_0309 = abs(yt - y_0309)
residual_0310 = abs(yt - y_0310)
residual_0311 = abs(yt - y_0311)
fig = plt.figure()
fun.display_result(Xt, residual_0309, fig, 331, 'Section_1020309', 'blue')
fun.display_result(Xt, residual_0310, fig, 332, 'Section_1020310', 'blue')
fun.display_result(Xt, residual_0311, fig, 333, 'Section_1020311', 'blue')

clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0310)
yt = clf.predict(Xt)
residual_0309 = abs(yt - y_0309)
residual_0310 = abs(yt - y_0310)
residual_0311 = abs(yt - y_0311)
fun.display_result(Xt, residual_0309, fig, 334, 'Section_1020309', 'blue')
fun.display_result(Xt, residual_0310, fig, 335, 'Section_1020310', 'blue')
fun.display_result(Xt, residual_0311, fig, 336, 'Section_1020311', 'blue')
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0311)
yt = clf.predict(Xt)
residual_0309 = abs(yt - y_0309)
residual_0310 = abs(yt - y_0310)
residual_0311 = abs(yt - y_0311)
fun.display_result(Xt, residual_0309, fig, 337, 'Section_1020309', 'blue')
fun.display_result(Xt, residual_0310, fig, 338, 'Section_1020310', 'blue')
fun.display_result(Xt, residual_0311, fig, 339, 'Section_1020311', 'blue')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()
