"""
Thesis Chapter 1 - Churn Estimation
Name : Shukhrat Khuseynov
ID   : 0070495
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def report (ytest, ypredict, detailed=False):
    """ reporting classification scores and details """
    
    accuracy = accuracy_score(ytest, ypredict)
    auroc = roc_auc_score(ytest, ypredict)
    conf = confusion_matrix(ytest, ypredict)
    
    print("\nAccuracy:", accuracy)
    print("\nAUROC:", auroc)
    print("\nConfusion matrix:")
    print(conf)
    
    if detailed == True:
        # plotting confusion matrix
        sns.heatmap(conf.T, square=True, annot=True, fmt='d', cbar=False)
        plt.axis('equal')
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()

        print("\nClassification report:")
        print(classification_report(ytest, ypredict))
    
    return (accuracy, auroc, conf)

def gridsearch (model, param, Xtrain, ytrain):
    """ implementing the process of GridSearchCV """
    
    grid = GridSearchCV(model, param, cv=5, scoring = 'roc_auc', refit=True, verbose=1)
    grid.fit(Xtrain, ytrain)

    print("\n", grid.best_score_)
    print("\n", grid.best_params_)
    print("\n", grid.best_estimator_)


# reading the data
df = pd.read_excel('mobile-churn-data.xlsx')

# checking variable types
print(df.info())

# checking whether there is any null element
print(df.isnull().values.any())

# detecting columns with diferent values for each row
print(df.loc[:, (df.nunique()==df.shape[0])].columns)

# dropping the id variable
df.drop(['user_account_id'], axis=1, inplace=True)

# churn distribution (pie chart)
exited = sum(df.churn == 1)
stayed = sum(df.churn == 0)

plt.pie([exited, stayed], labels=['Exited', 'Retained'], autopct='%1.1f%%')
plt.title('Churn distribution')
plt.axis('equal')
plt.show()


# initiating variables for the models
X = df.loc[:, df.columns != 'churn']
y = df.churn
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=0)

# scaling the data
scale = MinMaxScaler(feature_range=(0,1))
scale.fit(Xtrain)
Xtrain = scale.transform(Xtrain)
Xtest = scale.transform(Xtest)


# Models:

print("\n\nGaussian Naive Bayes classifier")
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(Xtrain, ytrain)
ypredict = nb.predict(Xtest)

nb_accuracy, nb_auroc, nb_conf = report(ytest, ypredict, detailed = False)
nb_fpr, nb_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nLogistic Regression")
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(solver='sag', max_iter=500)
reg.fit(Xtrain, ytrain)
ypredict = reg.predict(Xtest)

reg_accuracy, reg_auroc, reg_conf = report(ytest, ypredict, detailed = False)
reg_fpr, reg_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nK Nearest Neighbors classifier")
from sklearn.neighbors import KNeighborsClassifier

#param = {'n_neighbors': [10, 50, 100]}
#gridsearch(KNeighborsClassifier(), param, Xtrain, ytrain)
# choosing 50, more neighbors do not improve the model significantly

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(Xtrain, ytrain)
ypredict = knn.predict(Xtest)

knn_accuracy, knn_auroc, knn_conf = report(ytest, ypredict, detailed = False)
knn_fpr, knn_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nRandom Forest classifier")
from sklearn.ensemble import RandomForestClassifier

#param = {'n_estimators': [10, 50, 100, 200]}
#gridsearch(RandomForestClassifier(), param, Xtrain, ytrain)
# choosing 50, more estimators do not improve the model significantly

rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(Xtrain, ytrain) 
ypredict = rf.predict(Xtest)

rf_accuracy, rf_auroc, rf_conf = report(ytest, ypredict, detailed = False)
rf_fpr, rf_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nSupport Vector Machines classifier")
from sklearn.svm import SVC

#param = {'C': [9, 10, 11], 'gamma': ['auto'], 'kernel': ['rbf']}
#gridsearch(SVC(), param, Xtrain, ytrain)

svm = SVC(kernel='rbf', C=10, gamma='auto')
svm.fit(Xtrain, ytrain)
ypredict = svm.predict(Xtest)

svm_accuracy, svm_auroc, svm_conf = report(ytest, ypredict, detailed = False)
svm_fpr, svm_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nNeural Networks classifier")
from sklearn.neural_network import MLPClassifier

# tuned manually

nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=0)  
nn.fit(Xtrain, ytrain)
ypredict = nn.predict(Xtest)

nn_accuracy, nn_auroc, nn_conf = report(ytest, ypredict, detailed = False)
nn_fpr, nn_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nExtreme Gradient Booster classifier")
from xgboost import XGBClassifier

#param = {'n_estimators': [12, 13, 14]}
#gridsearch(XGBClassifier(), param, Xtrain, ytrain)

xgb = XGBClassifier(n_estimators=13, random_state=0)
xgb.fit(Xtrain, ytrain)
ypredict = xgb.predict(Xtest)

xgb_accuracy, xgb_auroc, xgb_conf = report(ytest, ypredict, detailed = False)
xgb_fpr, xgb_tpr, thresholds = roc_curve(ytest, ypredict)


# Plotting ROC curve:

plt.figure(figsize = (10,5))
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')

plt.plot(nb_fpr, nb_tpr, label = 'Naive Bayes: ' + str(round(nb_auroc, 3)))
plt.plot(reg_fpr, reg_tpr, label = 'Logistic Regression: ' + str(round(reg_auroc, 3)))
plt.plot(knn_fpr, knn_tpr, label = 'K Nearest Neighbors: ' + str(round(knn_auroc, 3)))
plt.plot(rf_fpr, rf_tpr, label = 'Random Forest: ' + str(round(rf_auroc, 3)))
plt.plot(svm_fpr, svm_tpr, label = 'Support Vector Machines: ' + str(round(svm_auroc, 3)))
plt.plot(nn_fpr, nn_tpr, label = 'Neural Networks: ' + str(round(nn_auroc, 3)))
plt.plot(xgb_fpr, xgb_tpr, label = 'XGBoost: ' + str(round(xgb_auroc, 3)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.50')
plt.legend()
plt.show()


# XGBoost seems to be slightly better than the others both in terms of accuracy and ROC.

xgb_accuracy, xgb_auroc, xgb_conf = report(ytest, ypredict, detailed = True)


# The end.
