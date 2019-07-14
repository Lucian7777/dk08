#coding=utf-8
import scipy.io as scio
from sklearn import svm
import numpy as np
import xlrd as xd
import matplotlib.pyplot as plt
import function as fun
from sklearn.externals import joblib
CLASSSNUM = 1440
#选取区段
SECTION = 101
#3.09数据处理以及时间数据初始化
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
fig = plt.figure()
fig2 = plt.figure()
fun.display_result(Xt, y_0310, fig2, 121, 'Section_' + str(SECTION) + '0310', 'blue', y_title='MTV')
fun.display_result(Xt, yt, fig2, 121, 'Section_' + str(SECTION) + '0310', 'red', y_title='MTV')
e_ = (yt - y_0310) / y_0310
fun.display_result(Xt, e_, fig2, 122, 'Section_' + str(SECTION) + '0310', 'blue')
# joblib.dump(clf, 'D:\python\svr_dk08\model\Svr.pkl')
# model = joblib.load('D:\python\svr_dk08\model\Svr.pkl')
#以0309为训练集作图结果
fun.display_result(Xt, y_0309, fig, 331, 'Section_' + str(SECTION) + '0309', 'blue')
fun.display_result(Xt, y_0310, fig, 332, 'Section_' + str(SECTION) + '0310', 'blue')
fun.display_result(Xt, y_0311, fig, 333, 'Section_' + str(SECTION) + '0311', 'blue')
fun.display_result(Xt, yt, fig, 331, 'Section_' + str(SECTION) + '0309', 'red')
fun.display_result(Xt, yt, fig, 332, 'Section_' + str(SECTION) + '0310', 'red')
fun.display_result(Xt, yt, fig, 333, 'Section_' + str(SECTION) + '0311', 'red')

clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0310)
yt = clf.predict(Xt)
#以0310为训练集作图结果
fun.display_result(Xt, y_0309, fig, 334, 'Section_' + str(SECTION) + '0309', 'blue')
fun.display_result(Xt, y_0310, fig, 335, 'Section_' + str(SECTION) + '0310', 'blue')
fun.display_result(Xt, y_0311, fig, 336, 'Section_' + str(SECTION) + '0311', 'blue')
fun.display_result(Xt, yt, fig, 334, 'Section_' + str(SECTION) + '0309', 'red')
fun.display_result(Xt, yt, fig, 335, 'Section_' + str(SECTION) + '0310', 'red')
fun.display_result(Xt, yt, fig, 336, 'Section_' + str(SECTION) + '0311', 'red')
clf = svm.SVR(C=9.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(Xt, y_0311)
yt = clf.predict(Xt)
print(yt)
#以0311为训练集作图结果
fun.display_result(Xt, y_0309, fig, 337, 'Section_' + str(SECTION) + '0309', 'blue')
fun.display_result(Xt, y_0310, fig, 338, 'Section_' + str(SECTION) + '0310', 'blue')
fun.display_result(Xt, y_0311, fig, 339, 'Section_' + str(SECTION) + '0311', 'blue')
fun.display_result(Xt, yt, fig, 337, 'Section_' + str(SECTION) + '0309', 'red')
fun.display_result(Xt, yt, fig, 338, 'Section_' + str(SECTION) + '0310', 'red')
fun.display_result(Xt, yt, fig, 339, 'Section_' + str(SECTION) + '0311', 'red')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
plt.show()






