import scipy.io as scio
import numpy as np


def display_result(time, value, figure_name, postion, title_name, col):
    ax = figure_name.add_subplot(postion)
    ax.set(xlim=[0, 1440], ylim=[value.min() * 1.5, value.max() * 1.5], title=title_name,
           ylabel='MTV', xlabel='time series')
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