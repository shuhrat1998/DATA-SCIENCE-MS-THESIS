"""
Thesis Chapter 3 - Sentiment Analysis
Name : Shukhrat Khuseynov
ID   : 0070495
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

def report2 (ytest, ypredict, graph = True, detailed=False):
    """ reporting classification scores and details """
    
    accuracy = accuracy_score(ytest, ypredict)
    # auroc = roc_auc_score(ytest, ypredict)
    conf = confusion_matrix(ytest, ypredict)
    
    print("\nAccuracy:", accuracy)
    # print("\nAUROC:", auroc)
    print("\nConfusion matrix:")
    print(conf)
    
    if graph == True:
        # plotting confusion matrix
        sns.heatmap(conf.T, square=True, annot=True, fmt='d', cbar=False)
        plt.axis('equal')
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
    
    if detailed == True:
        print("\nClassification report:")
        print(classification_report(ytest, ypredict))
    
    return (accuracy, 0, conf) #accuracy, auroc, conf

def gridsearch (model, param, Xtrain, ytrain, metric = 'roc_auc'):
    """ implementing the process of GridSearchCV """
    
    grid = GridSearchCV(model, param, cv=5, scoring = metric, refit=True, verbose=1)
    grid.fit(Xtrain, ytrain)

    print("\n", grid.best_score_)
    print("\n", grid.best_params_)
    print("\n", grid.best_estimator_)

def plot_wordcloud(df, title = None):
    stopwords = set(STOPWORDS)
    
    wd = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(" ".join(list(df))) # .generate(str(df))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wd)
    plt.show()


# reading the data
df = pd.read_csv('reviews.csv')

# checking variable types
#print(df.info())

# dropping unused columns
df = df.drop(df.columns[7:], axis=1)
df = df.drop(df.columns[:4], axis=1)

# rename columns
df.columns=['review', 'rating', 'recommend']
print(df.info())

# checking whether there is any null element
print(df.isnull().values.any())

# dropping the null values
df = df.dropna()
df.reset_index(drop=True, inplace=True)


# distribution of recommendations (pie chart)
pos = sum(df.recommend == 1)
neg = sum(df.recommend == 0)

plt.pie([pos, neg], labels=['Positive', 'Negative'], autopct='%1.1f%%')
plt.title('Recommendations')
plt.axis('equal')
#plt.savefig("Recom")
plt.show()


# distribution of ratings (pie chart)
r1 = sum(df.rating == 1)
r2 = sum(df.rating == 2)
r3 = sum(df.rating == 3)
r4 = sum(df.rating == 4)
r5 = sum(df.rating == 5)

plt.pie([r1, r2, r3, r4, r5], labels=['1', '2', '3', '4', '5'], autopct='%1.1f%%')
plt.title('Ratings')
plt.axis('equal')
#plt.savefig("Rank")
plt.show()


# preprocessing
for i in range(df.shape[0]):
    #if type(df.review[i]) == str:
    df.iloc[i,0] = re.sub('[^A-Za-z]+', ' ', df.iloc[i,0]).lower()

# tag cloud
plot_wordcloud(df.review[df.recommend == 1], title = "Positive Reviews")
plot_wordcloud(df.review[df.recommend == 0], title = "Negative Reviews")


# unigram counts (BoW)
vec = CountVectorizer(ngram_range=(1, 1), stop_words = {'english'})
vec.fit(df.review)
# print(vec.get_feature_names())
bow = vec.transform(df.review)

# bigram counts (BoW)
vec = CountVectorizer(ngram_range=(1, 2))
#vec.fit(df.review)
#bow2 = vec.transform(df.review)  # omitted

# unigram TF-IDF
vec = TfidfTransformer()
vec.fit(bow)
tfidf = vec.transform(bow)

# bigram TF-IDF
vec = TfidfTransformer()
#vec.fit(bow2)
#tfidf2 = vec.transform(bow2)    # omitted

# target
y1 = df.recommend
y2 = df.rating


""" Binary classification """

# initiating variables for the models
# try both BoW and TF-IDF:

Xtrain, Xtest, ytrain, ytest = train_test_split(bow, y1, test_size=0.20, random_state=0)
#Xtrain, Xtest, ytrain, ytest = train_test_split(tfidf, y1, test_size=0.20, random_state=0)


# Models:

print("\n\nGaussian Naive Bayes classifier")
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(Xtrain.toarray() , ytrain)
ypredict = nb.predict(Xtest.toarray())

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

#param = {'n_neighbors': [40, 50, 100]}
#gridsearch(KNeighborsClassifier(), param, Xtrain, ytrain)
# choosing 50, more neighbors do not improve the model significantly

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(Xtrain, ytrain)
ypredict = knn.predict(Xtest)

knn_accuracy, knn_auroc, knn_conf = report(ytest, ypredict, detailed = False)
knn_fpr, knn_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nRandom Forest classifier")
from sklearn.ensemble import RandomForestClassifier

#param = {'n_estimators': [100, 150]}
#gridsearch(RandomForestClassifier(), param, Xtrain, ytrain)
# choosing 100, more estimators do not improve the model significantly

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(Xtrain, ytrain) 
ypredict = rf.predict(Xtest)

rf_accuracy, rf_auroc, rf_conf = report(ytest, ypredict, detailed = False)
rf_fpr, rf_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nSupport Vector Machines classifier")
from sklearn.svm import SVC

#param = {'C': [100, 1000], 'gamma': ['auto'], 'kernel': ['rbf']}
#gridsearch(SVC(), param, Xtrain, ytrain)

C1 = 100       # for bow
C2 = 10000     # for tfidf

svm = SVC(kernel='rbf', C=C1, gamma='auto')
svm.fit(Xtrain, ytrain)
ypredict = svm.predict(Xtest)

svm_accuracy, svm_auroc, svm_conf = report(ytest, ypredict, detailed = False)
svm_fpr, svm_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nNeural Networks classifier")
from sklearn.neural_network import MLPClassifier

# tuned manually

Model1 = (5, 5, 5, 5, 5, 5)     # for bow
Model2 = (7, 7, 7, 7)           # for tfidf

nn = MLPClassifier(hidden_layer_sizes=Model1, max_iter=500, random_state=0)  
nn.fit(Xtrain, ytrain)
ypredict = nn.predict(Xtest)

nn_accuracy, nn_auroc, nn_conf = report(ytest, ypredict, detailed = False)
nn_fpr, nn_tpr, thresholds = roc_curve(ytest, ypredict)


print("\n\nExtreme Gradient Booster classifier")
from xgboost import XGBClassifier

#param = {'n_estimators': [900, 1000, 3000]}
#gridsearch(XGBClassifier(), param, Xtrain, ytrain)
# choosing 1000

xgb = XGBClassifier(n_estimators=1000, random_state=0)
xgb.fit(Xtrain, ytrain)
ypredict = xgb.predict(Xtest)

xgb_accuracy, xgb_auroc, xgb_conf = report(ytest, ypredict, detailed = False)
xgb_fpr, xgb_tpr, thresholds = roc_curve(ytest, ypredict)


# Plotting ROC curve:

plt.figure(figsize = (10,5))
plt.title('Receiver Operating Characteristic (ROC) curve using BoW/TF-IDF')
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


# nn_accuracy, nn_auroc, nn_conf = report(ytest, ypredict, detailed = True)


""" Multiclass classification """

# initiating variables for the models
# try both BoW and TF-IDF:

Xtrain, Xtest, ytrain, ytest = train_test_split(bow, y2, test_size=0.20, random_state=0)
#Xtrain, Xtest, ytrain, ytest = train_test_split(tfidf, y2, test_size=0.20, random_state=0)


# Models:

print("\n\nLogistic Regression")
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(solver='sag', max_iter=500)
reg.fit(Xtrain, ytrain)
ypredict = reg.predict(Xtest)

reg_accuracy, reg_auroc, reg_conf = report2(ytest, ypredict, graph = False, detailed = True)


print("\n\nSupport Vector Machines classifier")
from sklearn.svm import SVC

#param = {'C': [100, 1000, 10000], 'gamma': ['auto'], 'kernel': ['rbf']}
#gridsearch(SVC(), param, Xtrain, ytrain, metric = 'f1_macro')

C1 = 1000      # for bow
C2 = 10000     # for tfidf

svm = SVC(kernel='rbf', C=C1, gamma='auto')
svm.fit(Xtrain, ytrain)
ypredict = svm.predict(Xtest)

svm_accuracy, svm_auroc, svm_conf = report2(ytest, ypredict, graph = False, detailed = True)


print("\n\nNeural Networks classifier")
from sklearn.neural_network import MLPClassifier

# tuned manually

Model = (100)  # for both bow & tfidf

nn = MLPClassifier(hidden_layer_sizes=Model, max_iter=500, random_state=0)  
nn.fit(Xtrain, ytrain)
ypredict = nn.predict(Xtest)

nn_accuracy, nn_auroc, nn_conf = report2(ytest, ypredict, graph = False, detailed = True)


print("\n\nExtreme Gradient Booster classifier")
from xgboost import XGBClassifier

#param = {'n_estimators': [100, 150]}
#gridsearch(XGBClassifier(), param, Xtrain, ytrain, metric = 'f1_macro')

xgb = XGBClassifier(n_estimators=150, random_state=0)
xgb.fit(Xtrain, ytrain)
ypredict = xgb.predict(Xtest)

xgb_accuracy, xgb_auroc, xgb_conf = report2(ytest, ypredict, graph = False, detailed = True)

# The end.
