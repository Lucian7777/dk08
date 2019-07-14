import scipy.io as scio
import numpy as np

# coding=utf-8


def display_result(time, value, figure_name, postion, title_name, col, x_title='Time Series', y_title='Error'):
    ax = figure_name.add_subplot(postion)
    ax.set(xlim=[0, 1440], ylim=[value.min() * 0.9, value.max() * 1.1], title=title_name,
           ylabel=y_title, xlabel=x_title)
    ax.plot(time, value, color=col, linewidth=0.7)


def get_data_by_section(Section):
    data_path = "D:\matlab2016\\bin\\" + str(Section) + ".mat"
    data = scio.loadmat(data_path)
    data_ar = data.get('data')
    return data_ar


def split_by_day(data, data_x, index):
    y = np.zeros(1440)
    for i in data_x:
        y[i] = data[i + index]
    return y


def residual_data(num, section_data):
    y_base = np.zeros(num)
    for i in range(num):
        y_base[i] = section_data[0]
    y_re = section_data - y_base
    return y_re
