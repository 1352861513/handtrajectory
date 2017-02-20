import numpy as np
import os
import random

FRAME_COUNT = 10

def loadDataLabel(dir_name, shuffle=False, various=False):
    assert os.path.isdir(dir_name), "dir_name is not dir"
    dir = os.listdir(dir_name)
    dir.sort()
    len_dir = len(dir)
    datas = []
    labels = []
    for i in range(len_dir):
        f = open(dir_name + '/' + dir[i], "r")
        lines = f.readlines()
        for line in lines:
            words = line.split(' ')
            len_frame = (len(words) - 3) / 2
            if len_frame < FRAME_COUNT:
                continue
            temp = np.linspace(1, len_frame, FRAME_COUNT)
            index_x = np.zeros_like(temp, dtype=int)
            for k in range(len(temp)):
                index_x[k] = round(temp[k]) * 2
            # index_x = np.linspace(1, len_frame, FRAME_COUNT, dtype=int) * 2
            index_y = index_x + 1
            index_sample = np.concatenate((index_x,index_y))
            index_sample = np.sort(index_sample)

            temp = []
            for j in index_sample:
                temp.append(float(words[j]))

            data = np.asarray(temp, np.float32)
            label = int(words[1])

            # if label == 0:
            #     datas.append(data)
            #     labels.append(0)
            #     datas.append(horizontal(data))
            #     labels.append(0)
            #     datas.append(vertical(data))
            #     labels.append(0)
            #     datas.append(vertical(horizontal(data)))
            #     labels.append(0)
            # if label == 4:
            #     datas.append(data)
            #     labels.append(1)
            # if label == 5:
            #     datas.append(data)
            #     labels.append(2)

            datas.append(data)
            labels.append(label)

            if various:
                if label == 4:
                    continue
                elif label == 5:
                    continue
                elif label == 2:
                    var_label = 3
                elif label == 3:
                    var_label = 2
                else:
                    var_label = label
                hor_data = horizontal(data)
                ver_data = vertical(data)
                hv_data = vertical(hor_data)

                datas.append(hor_data)
                datas.append(ver_data)
                datas.append(hv_data)
                labels.append(var_label)
                labels.append(var_label)
                labels.append(label)
        f.close()

    if shuffle:
        print("Shuffling...")
        index = range(len(labels))
        random.shuffle(index)
        xx = []
        yy = []
        for i in range(len(labels)):
            xx.append(datas[index[i]])
            yy.append(labels[index[i]])
        datas = xx
        labels = yy
    return np.asarray(datas, np.float32), np.asarray(labels, np.float32)

def horizontal(data):
    length = len(data)
    assert (length % 2) == 0
    index = np.arange(length / 2) * 2
    result = np.zeros_like(data)
    for i in range(length / 2):
        result[index[i]] = data[index[i]] * -1
        result[index[i] + 1] = data[index[i] + 1]
    return result


def vertical(data):
    length = len(data)
    assert (length % 2) == 0
    index = np.arange(length / 2) * 2
    result = np.zeros_like(data)
    for i in range(length / 2):
        result[index[i]] = data[index[i]]
        result[index[i] + 1] = data[index[i] + 1] * -1
    return result
