from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import pandas as pd

# data = datasets.load_breast_cancer()
# X,y = data.data,data.target
# p = type(data)
# X_train, X_test, y_train,y_test = train_test_split(
#     X,y,test_size=0.2,random_state=1234
# )


dt_train = pd.read_csv("training.csv", skiprows=1, header=None)
dt_test = pd.read_csv("test.csv", skiprows=1, header=None)
dt_cost = pd.read_csv("cost.csv", header=None)
X_train = (dt_train.iloc[:,:-1]).to_numpy()
y_train = (dt_train.iloc[:,-1]).to_numpy()
X_test = (dt_test.iloc[:,:-1]).to_numpy()
y_test = (dt_test.iloc[:,-1]).to_numpy()
cost = (dt_cost.iloc[:,0]).to_numpy()
cost = np.append(cost, 1)


cost = None

clf = DecisionTree()
clf.fit(X_train,y_train,cost)
predictions = clf.predict(X_train)


# costs = clf.predict_cost(X_train)
# costs = np.array(costs)
# costs_per_class=[]
# for label in np.unique(y_train):
#     indices = np.where(y_train == label)
#     a = np.sum(costs[indices])
#     b = len(indices[0])
#     costs_per_class.append(np.sum(costs[indices]) / len(indices[0]))




clf.print_tree()



def total_accuracy(y_test,y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


def confusion_matrix(y_true, y_pred):
    K = 3  # number of classes
    result = np.zeros((K, K))
    for i in range(len(y_true)):
        result[y_true[i]-1][y_pred[i]-1] += 1
    return result


def measuresAll(cm):
    sensitivity = []
    specificity = []
    precision = []
    accuracy = []
    fscore = []
    dice = []
    n = len(y_train)
    for i in range(len(cm)):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[:, i]) - tp
        tn = 0
        for j in range(len(cm)):
            if j != i:
                tn += sum(cm[j]) - cm[j][i]  # sum of true negatives for other classes
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        prec = tp / (tp + fp)
        acc = (tp + tn) / n
        fs = (2*prec*sens) / (prec+sens)
        d = (2*tp) / (2*tp + fp + fn)
        sensitivity.append(sens)
        specificity.append(spec)
        precision.append(prec)
        accuracy.append(acc)
        fscore.append(fs)
        dice.append(d)

    return sensitivity, specificity,precision,accuracy,fscore,dice




matrix = confusion_matrix(y_train,predictions)
sensitivity , specificity,precision,accuracy,fscore,dice= measuresAll(matrix)
acc = total_accuracy(y_test,predictions)
clf.print_used_nodes()
print(acc)

y = y_train
unique_labels = np.unique(y)
hist = np.bincount(y)
n_y = len(y)
hist = hist[1:]


avg_sensitivity = np.sum(hist * sensitivity) / n_y
avg_specificity = np.sum(hist * specificity) / n_y
avg_precision = np.sum(hist * precision) / n_y
avg_accuracy = np.sum(hist * accuracy) / n_y
avg_fscore = np.sum(hist * fscore) / n_y
avg_dice = np.sum(hist * dice) / n_y


print(sensitivity , specificity,precision,accuracy,fscore,dice)


