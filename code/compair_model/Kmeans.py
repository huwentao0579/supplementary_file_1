from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc, f1_score
import numpy as np
from scipy import stats
import time

def load_data(file):
    with open(file) as f:
        lines = f.readlines()
    dataSet = []
    for line in lines:
        row = []
        line = line.split(',')
        for l in line:
            l = float(l)
            row.append(l)
        dataSet.append(row)
    return np.mat(dataSet)

def f1_score(y_true, y_score):
    precision, recall, t = precision_recall_curve(1-y_true, 1-y_score)
    f1 = 2*(precision * recall)/(precision + recall)

    where_are_NaN = np.isnan(f1)
    f1[where_are_NaN] = 0

    f1_index = np.where(f1 == np.max(f1))[0][0]
    threshold = t[f1_index]

    a= 1-y_true
    b = 1-y_score
    tp = []
    for i in b:
        if i < threshold:
            tp.append(0)
        else:
            tp.append(1)
    import pandas as pd
    tt = pd.DataFrame(tp)
    tt.to_csv('tp2.csv')
    # return auc(recall, precision)
    return f1.max()


def ge_labels(data):

    a = np.ones(len(data))
    # outliers = [4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999]
    outliers = []
    # outliers = [212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234]
    for i in outliers:
        a[i] = 0
    return a
def computer_likehood(da, label, data, labels_):
    Mean = np.mean(data[labels_ == label], axis=0)
    Var = np.var(data[labels_ == label], axis=0)
    tp_likehood = 0
    for i in range(da.size):
        tp = stats.norm.pdf(da[0,i],Mean[0,i],Var[0,i]+ 0.000000001)
        if tp > 1:
            tp = 1
        if tp == 0:
            continue
        tp = np.log(tp)
        tp_likehood += tp
    tp_likehood = tp_likehood/ da.size
    tp_likehood = np.exp(tp_likehood)
    return tp_likehood


def computer_score(da,center):
    distance = -(abs(da - center) / center ).sum()
    score = np.exp(distance)
    return score


if __name__ == '__main__':
    file = './datasets/test_100.csv'
    data = load_data(file)
    time1 = time.time()
    estimator = KMeans(1, max_iter=100).fit(data)
    time2 = time.time()
    labels_ = estimator.labels_
    centers = estimator.cluster_centers_
    predictors = []
    for i in range(len(data)):
        label = labels_[i]
        center = centers[label]
        da = data[i]
        score = computer_score(da,center)
        # likehood = computer_likehood(da,label,data,labels_)
        predictors.append(score)


    y_true = ge_labels(data)

    y_pre = np.array(predictors)
    f1 = f1_score(y_true,y_pre)

    print(time2-time1)
    print(f1)









