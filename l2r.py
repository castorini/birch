import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import svm
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import operator
def load_data(topKSent):
    data= pd.read_csv("qa.bert.1000.scores", sep=' ', header=None)
    df= pd.DataFrame(data)

    X= df[range(3, 3+topKSent)].values
    y= df[[2]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X, y

def SVM(X_train, X_test, y_train, y_test):

    svc = svm.SVC(kernel='linear')

    svc.fit(X_train, y_train)

    predn=svc.predict(X_test)
    for score in predn:
        print score

    print('The accuracy of the model is',metrics.accuracy_score(predn,y_test))

def LR(X_train, X_test, y_train, y_test):

    clf = LogisticRegression().fit(X_train, y_train)

    scores = clf.predict_proba(X_train)
    pos_scores = []
    for pair in scores:
        pos_scores.append(pair[1])
    # print clf.coef_
    # print('The accuracy of the model is',metrics.accuracy_score(predn,y_test))
    return pos_scores

def rerank(pos_scores):
    q_list = []
    d_list = []
    with open("qa.bert.1000.scores") as qF:
        for line in qF:
            q_list.append(line.strip().split()[0])
            d_list.append(line.strip().split()[1])
    score_dict = defaultdict(dict)
    for q,d,score in zip(q_list, d_list, pos_scores):
        score_dict[q][d] = score
    for q in score_dict:
        doc_score_dict = score_dict[q]
        doc_score_dict = sorted(doc_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_score_dict:
            print q, 'Q0', doc, rank, score, 'BM25'
            rank+=1


def main():
    X, y = load_data(5)
    pos_scores = LR(X, X, y, y)
    # rerank(pos_scores)

if __name__ == "__main__":
    main()