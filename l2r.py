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
import xgboost as xgb
from xgboost import DMatrix


def load_data(topKSent):
    max_len = 0
    with open("mb.robust04.para1.scores") as mbF:
        for line in mbF:
            length = len(line.strip().split())
            max_len = max(length, max_len)
    data= pd.read_csv("mb.robust04.para1.scores", sep=' ', header=None,
        names=list(range(max_len)))
    df= pd.DataFrame(data)
    
    # print (df)
    df.fillna(0)
    
    X= df[range(3, 3+topKSent)].values
    y= df[[2]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X, y

def SVM(X_train, X_test, y_train, y_test):

    svc = svm.SVC(kernel='linear')

    svc.fit(X_train, y_train)

    predn=svc.predict(X_test)
    for score in predn:
        print(score)

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

def LambdaMART(X_train, X_test, y_train, y_test):

    train_dmatrix = DMatrix(X_train, X_test)
    params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
               'min_child_weight': 0.1, 'max_depth': 6}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4)
    pred = xgb_model.predict(train_dmatrix)
    print(pred)
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
            print(q, 'Q0', doc, rank, score, 'BM25')
            rank+=1


def main():
    X, y = load_data(3)
    # pos_scores = LR(X, X, y, y)
    # rerank(pos_scores)
    LambdaMART(X,X,y,y)

if __name__ == "__main__":
    main()