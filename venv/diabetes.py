import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#read the dataset
diabetes_data = pd.read_csv('diabetes.csv')
print(len(diabetes_data))
print(diabetes_data.head())

#split dataset
# x is the columns
X = diabetes_data.values[:, 0:]
# y is the last column which is the result
y = diabetes_data.values[:, 8]
train, X_test, outcome, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(train, outcome, random_state=2,test_size=0.25)

#visualization
# diabetes_data.hist(bins=50, figsize=(20, 15))
# plt.show()


#replace zeros and null
# zeros_null = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
#change all zeros and null with median of those columns
# for column in zeros_null:
#     diabetes_data[column] = diabetes_data[column].replace(0, np.NaN)
#     median = int(diabetes_data[column].mean(skipna=True))
#     diabetes_data[column] = diabetes_data[column].replace(np.NaN, median)

#standardization
# from sklearn.preprocessing import StandardScaler
# rescaledX = StandardScaler()
# X_train = rescaledX.fit_transform(X_train)
# X_test = rescaledX.transform(X_test)

# knn-algorithm
# input: current vector in test_images, train_images, train_labels and k
def classify(inX, dataSet, labels, k):
    # get how many rows in traindata
    dataSetSize = dataSet.shape[0]
    # get the distance
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    seDiffMat = diffMat ** 2
    seDistances = seDiffMat.sum(axis = 1)
    distances = seDistances ** 0.5
    # sort by distance
    sortedDistIndicies = distances.argsort()
    classCount={}
    # get kth shortest distance images
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #find the classification that appear mostly
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# def check(k):
#     m = y_test.shape[0]
#     resultList = []
#     errorCount = 0
#     # 对每一个测试数据向量进行分类，并记录结果
#     for i in range(m):
#         currRes = classify(X_test[i], X_train, y_train, k)
#         resultList.append(int(currRes))
#         print ("the classifier came back with: %d, the real answer is: %d" % (int(currRes), y_test[i]))
#         if (int(currRes) != y_test[i]):
#             errorCount += 1.0
#         print ("\nthe total number of errors is: %d" % errorCount)
#         print(i)
#         print("######################################")
#     print("\nthe total error rate is: %f" % (errorCount / float(m)))
#     cm = confusion_matrix(y_test, resultList)
#     print(cm)

def runKnn(k):
    m = y_test.shape[0]
    errorCount = 0
    for i in range(m):
        currRes = classify(X_test[i], X_train, y_train, k)
        resultList.append(int(currRes))
        if (int(currRes) != y_test[i]):
            errorCount += 1.0
        print(i)
    return (errorCount / float(m))

def diffK():
    k_score = []
    test_range = X_valid.shape[0]
    for k in range(1, 33 + 1):
        print("k = {} Training.".format(k))
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predict_result = knn.predict(X_valid[:test_range])
        predict_score = accuracy_score(y_valid[:test_range], predict_result)
        k_score.append(predict_score)
        end = time.time()
        print("Score: {}.".format(predict_score))
        print("Complete time: {} Secs.".format(end - start))
    print(k_score)
    plt.plot(range(1, 33 + 1), k_score)
    plt.xlabel('k')
    plt.ylabel('k_score')
    plt.show()

def predict():
    k = 10
    print("k = {} Training.".format(k))
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predict_result = knn.predict(X_test)
    end = time.time()
    print("Complete time: {} Secs.".format(end - start))
    print(accuracy_score(y_test, predict_result))
    cm = confusion_matrix(y_test, predict_result)
    print(cm)


if __name__ == '__main__':
    # start = time.clock()
    # diffK()
    # end = time.clock()
    # print ('Time used: {}'.format(end - start))
    predict()