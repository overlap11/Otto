import pandas as pd
import numpy as np
import scipy as sp
import time
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

def stack_models(X_train, y_train, X_test, y_test, models, n_class, n_fold=10, random_state=123):

    folds = KFold(len(X_train), n_folds = n_fold, shuffle=True, random_state=random_state)

    for k, (train_index, valid_index) in enumerate(folds):
        print "Stacking fold :", k+1
        X_train_tmp = X_train[train_index,:]
        y_train_tmp = y_train[train_index]
        X_valid = X_train[valid_index,:]
        y_valid = y_train[valid_index]

        y_train_bin = LabelBinarizer().fit_transform(y_train_tmp)
        preds = []
        for clf in models:
            pred_valid = np.zeros((len(X_valid),n_class))
            for i in range(n_class):
                clf.fit(X_train_tmp, y_train_bin[:,i])
                pred_valid[:,i] = clf.predict_proba(X_valid)[:,1]
            preds.append(pred_valid)

        if k == 0:
            X_train2 = X_valid
            y_train2 = y_valid
            X_train_pred = preds
            index = valid_index
        else:
            X_train2 = np.r_[X_train2, X_valid]
            y_train2 = np.r_[y_train2, y_valid]
            X_train_pred = [ np.r_[X_train_pred[i], preds[i]] for i in range(len(models))]
            index = np.r_[index, valid_index]
    X_train_pred = [pred[np.argsort(index),:] for pred in X_train_pred]

    y_train_bin = LabelBinarizer().fit_transform(y_train)
    preds = []
    for clf in models:
        pred_test = np.zeros((len(X_test),n_class))
        for i in range(n_class):
            clf.fit(X_train, y_train_bin[:,i])
            pred_test[:,i] = clf.predict_proba(X_test)[:,1]
        preds.append(pred_test)

    X_test_pred = preds
    return([X_train_pred, X_test_pred])

def disp_performance(X_train, y_train, X_test, y_test):
    clf = MultinomialNB()
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "MultinomialNB", log_loss(y_test, clf.predict_proba(X_test))

    clf = LogisticRegression()
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "LogisticRegression", log_loss(y_test, clf.predict_proba(X_test))

    clf = RandomForestClassifier(n_estimators=100)
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "RandomForestClassifier", log_loss(y_test, clf.predict_proba(X_test))

    clf = SGDClassifier(loss="log")
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "SGDClassifier", log_loss(y_test, clf.predict_proba(X_test))

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "ExtraTreesClassifier", log_loss(y_test, clf.predict_proba(X_test))

    clf = AdaBoostClassifier(n_estimators=100)
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "AdaBoostClassifier", log_loss(y_test, clf.predict_proba(X_test))

    clf = DecisionTreeClassifier()
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "DecisionTreeClassifier", log_loss(y_test, clf.predict_proba(X_test))

    clf = KNeighborsClassifier(n_neighbors=10, algorithm='brute')
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "KNeighborsClassifier", log_loss(y_test, clf.predict_proba(X_test))

    clf = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6)
    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)
    print "XGBClassifier", log_loss(y_test, clf.predict_proba(X_test))

def main():
    X = pd.read_csv('data/train.csv')
    X = X.drop("id",axis=1)
    y = X.target
    y = LabelEncoder().fit_transform(y)
    X = X.drop("target",axis=1)
    X = X.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print "----------------------------------------------------------"
    print "Layer1"
    disp_performance(X_train, y_train, X_test, y_test)

    print "----------------------------------------------------------"
    print "Layer2"

    models = [CalibratedClassifierCV(MultinomialNB(),method="isotonic", cv=5),
              CalibratedClassifierCV(LogisticRegression(), method="isotonic", cv=5),
              CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, n_jobs=10), method="isotonic", cv=5),
              CalibratedClassifierCV(SGDClassifier(loss="log"), method="isotonic", cv=5),
              CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=100), method="isotonic", cv=5),
              CalibratedClassifierCV(AdaBoostClassifier(n_estimators=100), method="isotonic", cv=5),
              CalibratedClassifierCV(DecisionTreeClassifier(), method="isotonic", cv=5),
              CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=10, algorithm='brute'), method="isotonic", cv=5),
              CalibratedClassifierCV(XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6), method="isotonic", cv=5)]

    start = time.time()
    X_train_pred, X_test_pred = stack_models(X_train, y_train, X_test, y_test, models, n_class=9, n_fold=10, random_state=123)
    print "time =", time.time() - start

    X_train2 = X_train_pred[0]
    X_test2 = X_test_pred[0]
    for i in range(1,len(models)):
        X_train2 = np.c_[X_train2, X_train_pred[i]]
        X_test2 = np.c_[X_test2, X_test_pred[i]]

    disp_performance(X_train2, y_train, X_test2, y_test)

    print "----------------------------------------------------------"
    print "Layer3"

    models = [CalibratedClassifierCV(MultinomialNB(),method="isotonic", cv=5),
              CalibratedClassifierCV(LogisticRegression(), method="isotonic", cv=5),
              CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, n_jobs=10), method="isotonic", cv=5),
              CalibratedClassifierCV(SGDClassifier(loss="log"), method="isotonic", cv=5),
              CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=100), method="isotonic", cv=5),
              CalibratedClassifierCV(AdaBoostClassifier(n_estimators=100), method="isotonic", cv=5),
              CalibratedClassifierCV(DecisionTreeClassifier(), method="isotonic", cv=5),
              CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=10, algorithm='brute'), method="isotonic", cv=5),
              CalibratedClassifierCV(XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6), method="isotonic", cv=5)]

    start = time.time()
    X_train_pred2, X_test_pred2 = stack_models(X_train2, y_train, X_test2, y_test, models, n_class=9, n_fold=10, random_state=123)
    print "time =", time.time() - start

    X_train3 = X_train_pred2[0]
    X_test3 = X_test_pred2[0]
    for i in range(1,len(models)):
        X_train3 = np.c_[X_train3, X_train_pred2[i]]
        X_test3 = np.c_[X_test3, X_test_pred2[i]]

    X_train_tmp = X_train_pred[0]
    X_test_tmp = X_test_pred[0]
    for i in range(1,len(models)):
        X_train_tmp = np.c_[X_train_tmp, X_train_pred[i]]
        X_test_tmp = np.c_[X_test_tmp, X_test_pred[i]]

    X_train3 = np.c_[X_train_tmp, X_train3]
    X_test3 = np.c_[X_test_tmp, X_test3]

    X_train3 = np.c_[X_train, X_train3]
    X_test3 = np.c_[X_test, X_test3]

    disp_performance(X_train3, y_train, X_test3, y_test)

if __name__ == "__main__":
    main()

