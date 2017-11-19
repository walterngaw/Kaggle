
import DataModel
from sklearn import cross_validation
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np

def findBestNeighborsCount(x_train, x_cv, y_train, y_cv, kFrom, kTo, steps):
    kVec = np.linspace(kFrom, kTo, steps, dtype=int)

    bestAcc = 0.0
    bestK = 0

    for k in kVec:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(x_train, y_train)
        acc_cv = accuracy_score(y_cv, knn.predict(x_cv))
        print("Accuracy: {} with {} neighbors".format(acc_cv, k))
        if acc_cv > bestAcc:
            bestAcc = acc_cv
            bestK = k

    return (bestK, bestAcc)

dl = DataModel.DataLoader()

x, y = dl.loadTrainData("data/train.csv")

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)


(k, acc) = findBestNeighborsCount(x_train, x_cv, y_train, y_cv, 50, 200, 50)

print("Best k: {} with accuracy: {}".format(k, acc))
