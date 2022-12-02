import re
import numpy as np
import pandas as pd
import pandas_profiling as ppf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
import sklearn
import warnings
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# from __future__ import division
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
from scipy import interp
from datetime import datetime
warnings.filterwarnings("ignore")





'''
Dataset Preprocessing:
1. Remove HTML tags.
2. Fill in missing values.
3. Get hint number.
4. Get the number of similarity question with different difficulty.
5. Label Encode the "topic" feature to 71 new features.
6. Label Encode the Target Value (Difficulty Tag)
'''
# preprocess question_text
def remove_html_tags(text):
    patterns_rep_with_space = [r'&nbsp',r'\t', r'&lt',r',']
    patterns_rep_without_space = [r'<.*?>', r'\n', r';', r'&quot', r']']
    for pattern in patterns_rep_with_space:
        p = re.compile(pattern)
        text = p.sub(" ", text)
    for pattern in patterns_rep_without_space:
        p = re.compile(pattern)
        text = p.sub("", text)
    text = re.sub(r'\s{2,}', '', text)
    return text

# read clawer data
df = pd.read_csv('dataset/data_raw.csv')

# init text_length column = 0
df['question_text_length'] = 0
# loop question text to remove html tags & save the text length
for i in range(len(df['question_text'])):
    df['question_text'][i] = remove_html_tags(df['question_text'][i]).split('Example')[0]
    df['question_text_length'][i] = len(df['question_text'][i])


'''
Fill in the missing values.
'''
# init variable
sumOfEasy = 0
sumOfMedium = 0
sumOfHard = 0
countOfEasy = 0
countOfMedium = 0
countOfHard = 0
# calculate average length of each type question
for i in range(len(df['difficulty'])):
    if (df['difficulty'][i] == 'Easy'):
        countOfEasy = countOfEasy + 1
        sumOfEasy += df['question_text_length'][i]
    elif (df['difficulty'][i] == 'Medium'):
        countOfMedium = countOfMedium + 1
        sumOfMedium += df['question_text_length'][i]
    else:
        countOfHard = countOfHard + 1
        sumOfHard += df['question_text_length'][i]
averageEasy = sumOfEasy/countOfEasy
averageMedium = sumOfMedium/countOfMedium
averageHard = sumOfHard/countOfHard
for i in range(len(df['question_text'])):
    if (df['question_text'][i] == '1'):
        if (df['difficulty'][i] == 'Easy'):
               df['question_text_length'][i] = averageEasy
        elif (df['difficulty'][i] == 'Medium'):
               df['question_text_length'][i] = averageMedium
        else:
              df['question_text_length'][i] = averageHard


'''
Get the hint number.
'''
hints_raw = df['hints'].values
df['hint_number'] = 0
def preprocess_hints(hints):
    for i in range(len(hints)):
           if (len(hints[i]) > 2):
                df['hint_number'][i] = len(hints[i].split(", \'"))
preprocess_hints(hints_raw)

'''
Get the number of similarity question with different difficulty.
'''
df['simiquestions_easy'] = 0
df['simiquestions_medium'] = 0
df['simiquestions_hard'] = 0
def get_similar_question_slugs(similar_questions):
    for i in range(len(similar_questions)):
        df['simiquestions_easy'][i] = len(df['similar_questions'][i].split("'difficulty': 'Easy'")) - 1
        df['simiquestions_medium'][i] = len(df['similar_questions'][i].split("'difficulty': 'Medium'")) - 1
        df['simiquestions_hard'][i] = len(df['similar_questions'][i].split("'difficulty': 'Hard'")) - 1
get_similar_question_slugs(df['similar_questions'])


'''
Label Encode the "topic" feature to 71 new features.
If the value of new features equal to 1, then it means that question has this topic tag.
If the value of new features equal to 0, then it means the question don't have this topic tag.
'''

tag = ['Array', 'String', 'Hash Table', 'Dynamic Programming',
       'Math', 'Sorting', 'Greedy', 'Depth-First Search', 'Database',
       'Breadth-First Search', 'Tree', 'Binary Search', 'Matrix', 'Binary Tree',
      'Two Pointers', 'Bit Manipulation', 'Stack', 'Heap (Priority Queue)', 'Design', 'Graph',
      'Prefix Sum', 'Simulation', 'Backtracking', 'Counting', 'Sliding Window',
      'Union Find', 'Linked List', 'Ordered Set', 'Monotonic Stack', 'Enumeration', 'Recursion',
      'Trie', 'Divide and Conquer', 'Binary Search Tree', 'Bitmask', 'Queue', 'Memoization',
      'Geometry', 'Segment Tree', 'Topological Sort', 'Hash Function', 'Game Theory', 'Binary Indexed Tree',
      'Number Theory', 'Interactive', 'String Matching', 'Rolling Hash', 'Shortest Path', 'Data Stream',
      'Combinatorics', 'Randomized', 'Monotonic Queue', 'Brainteaser', 'Merge Sort', 'Iterator', 'Concurrency',
      'Doubly-Linked List', 'Probability and Statistics', 'Quickselect', 'Bucket Sort', 'Suffix Array',
      'Minimum Spanning Tree', 'Counting Sort', 'Shell', 'Line Sweep', 'Reservoir Sampling', 'Eulerian Circuit',
      'Radix Sort', 'Strongly Connected Component', 'Rejection Sampling', 'Biconnected Component'
      ]
for i in tag:
    df.insert(df.shape[1], i, 0)
    for j in range(0, len(df)):
        if (('\'' + i + '\'') in df['topic_tagged_text'][j]):
            df[i][j] = 1


# Convert string data to float data
df['success_rate'] = df['success_rate'].str.strip("%").astype(float)

'''
Label Encode the Target Value (Difficulty Tag)
'''
for i in range(0, len(df)):
    if (df['difficulty'][i] == 'Easy'):
        df['difficulty'][i] = 0
    elif (df['difficulty'][i] == 'Medium'):
        df['difficulty'][i] = 1
    else:
        df['difficulty'][i] = 2
df['difficulty'] = df['difficulty'].astype('int')


'''
Cleaning Dataset：
This part is mainly to lose the outliers in the data set, and carry out feature amplification for the obscure 
feature data in the data set, so as to improve the model performance.
'''
df.drop(df[(df['success_rate']>70) & (df['difficulty'] == 2)].index,inplace=True)
df.drop(df[(df['success_rate']>80) & (df['difficulty'] == 1)].index,inplace=True)
df.drop(df[(df['success_rate']<10) & (df['difficulty'] == 1)].index,inplace=True)
df.drop(df[(df['success_rate']<20) & (df['difficulty'] == 0)].index,inplace=True)
df.drop(df[(df['likes']<5) & (df['difficulty'] == 0)].index,inplace=True)
df.drop(df[(df['dislikes']>10000) & (df['difficulty'] == 0)].index,inplace=True)
p, q= df[(df.iloc[:,4] == 1) & (df.iloc[:,5] > 50)], df[(df.iloc[:,4] == 2) & (df.iloc[:,5] > 50)]
p.iloc[:,12], q.iloc[:,12]= (100 * p.iloc[:,12]).astype(int), (100 * q.iloc[:,12]).astype(int)
df[(df.iloc[:,4] == 1) & (df.iloc[:,5] > 50)],df[(df.iloc[:,4] == 2) & (df.iloc[:,5] > 50)]=p,q
# Remove the feature information which is not relevant to the result
df = df.drop(labels=['question_id','question_title','question_slug','question_text','hints',
                     'similar_questions','topic_tagged_text', 'company_tags'],axis=1)
df.reset_index(drop=True, inplace=True)

'''
Split Dataset to Training Dataset and Test Dataset
'''
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


'''
LR Cross Validation Select C
We adjust the hyperparameter C which determines the strength of the regularisation in terms of L2 penalty.
Range: [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
Method: 5-Fold Cross Validation
Result: C = 0.1
'''
mean_error=[]; std_error=[]
Ci_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
logC = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
for Ci in Ci_range:
    model = LR(multi_class="multinomial", solver="newton-cg", max_iter=1000, C = Ci)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train_lr, X_test_lr = X.iloc[train], X.iloc[test]
        y_train_lr, y_test_lr = y.iloc[train], y.iloc[test]
        model.fit(X_train_lr, y_train_lr)
        ypred = model.predict(X_test_lr)
        temp.append(mean_squared_error(y_test_lr,ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(logC,mean_error,yerr=std_error)
plt.xlabel('lg(C)'); plt.ylabel('Mean square error')
plt.xlim((-6,3))
plt.show()
'''
LR model evaluation result
'''
print("*****Logistic Regression Model Evaluation Result*****")
lr = LR(multi_class="multinomial", solver="newton-cg", max_iter=1000, C = 0.1)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred , digits=4))




'''
kNN Cross Validation Select k:
We adjust the parameter k, which is the number of neighbors to use.
Range: (1 , 20) only odd number
Method: 5-Fold Cross Validation
Result: k = 5
'''
mean_error=[]; std_error=[]
K_range = [1,3,5,7,9,11,13,15,17,19]
for k in K_range:
    model = KNeighborsClassifier(n_neighbors=k, p=1)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train_knn, X_test_knn = X.iloc[train], X.iloc[test]
        y_train_knn, y_test_knn = y.iloc[train], y.iloc[test]
        model.fit(X_train_knn, y_train_knn)
        ypred = model.predict(X_test_knn)
        temp.append(mean_squared_error(y_test_knn,ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(K_range,mean_error,yerr=std_error)
plt.xlabel('K'); plt.ylabel('Mean square error')
plt.xlim((1, 20))
plt.show()
'''
kNN model evaluation result
'''
print("****************kNN Evaluation Result****************")
knn = KNeighborsClassifier(n_neighbors=5, p=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred , digits=4))



'''
Decision Tree Cross Validation Select max_depth:
We adjust the maximum depth of the tree (max_depth)
Range: (1 , 10)
Method: 5-Fold Cross Validation
Result: max_depth = 7
'''
mean_error=[]; std_error=[]
depth = []
for i in range(1,11):
    depth.append(i)
for k in depth:
    model = DecisionTreeClassifier(max_depth = k)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train_dt, X_test_dt = X.iloc[train], X.iloc[test]
        y_train_dt, y_test_dt = y.iloc[train], y.iloc[test]
        model.fit(X_train_dt, y_train_dt)
        ypred = model.predict(X_test_dt)
        temp.append(mean_squared_error(y_test_dt,ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(depth,mean_error,yerr=std_error)
plt.xlabel('Depth'); plt.ylabel('Mean square error')
plt.xlim((1, 10))
plt.show()
'''
Decision Tree model evaluation result
'''
print("********Decision Tree Model Evaluation Result********")
dt = DecisionTreeClassifier(max_depth = 7)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred , digits=4))



'''
Random Forest Cross Validation Select n_estimators:
First adjust for the main parameter (n_estimators), which refers to the number of random forest spanning trees.
Range: (1 , 50)
Method: 5-Fold Cross Validation
Result: n_estimators = 40
'''
mean_error=[]; std_error=[]
n_estimators = []
for i in range(1,50):
    n_estimators.append(i)
for k in n_estimators:
    model = RandomForestClassifier(random_state = 0, n_estimators=k)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train_rf, X_test_rf = X.iloc[train], X.iloc[test]
        y_train_rf, y_test_rf = y.iloc[train], y.iloc[test]
        model.fit(X_train_rf, y_train_rf)
        ypred = model.predict(X_test_rf)
        temp.append(mean_squared_error(y_test_rf,ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(n_estimators,mean_error,yerr=std_error)
plt.xlabel('n_estimators'); plt.ylabel('Mean square error')
plt.xlim((1, 50))
plt.show()

'''
Random Forest Cross Validation Select max_depth:
Then we adjust the maximum depth of the tree (max_depth)
Range: (1 , 25)
Method: 5-Fold Cross Validation
Result: max_depth = 15
'''
mean_error=[]; std_error=[]
depth = []
for i in range(1,26):
    depth.append(i)
for k in depth:
    model = RandomForestClassifier(random_state = 0, n_estimators=40, max_depth = k)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train_rf1, X_test_rf1 = X.iloc[train], X.iloc[test]
        y_train_rf1, y_test_rf1 = y.iloc[train], y.iloc[test]
        model.fit(X_train_rf1, y_train_rf1)
        ypred = model.predict(X_test_rf1)
        temp.append(mean_squared_error(y_test_rf1,ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(depth,mean_error,yerr=std_error)
plt.xlabel('Depth'); plt.ylabel('Mean square error')
plt.xlim((1, 25))
plt.show()


'''
RF model evaluation result:
'''
print("********Random Forest Model Evaluation Result********")
rfc = RandomForestClassifier(random_state = 0, n_estimators=40, max_depth = 15)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred , digits=4))

'''
Dummy (Baseline) evaluation result:
'''
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
print(classification_report(y_test, y_pred , digits=4))


'''
Evaluation Result:
1. P/R Curve.
2. ROC Curve.
3. Confusion Matrixs.
'''
mpl.rcParams.update(mpl.rcParamsDefault)
n_classes = 3
simpleLabels = ["lr", "rf", "dt", "knn", "dummy"]
# For each class
precision = dict()
recall = dict()
average_precision = dict()
# score for each model
y_score_lr = lr.predict_proba(X_test)
y_score_rf = rfc.predict_proba(X_test)
y_score_dt = dt.predict_proba(X_test)
y_score_knn = knn.predict_proba(X_test)
y_score_dummy = dummy.predict_proba(X_test)
# Binarize the output
y_test_plot = label_binarize(y_test, classes=[0, 1, 2])
score = {"lr":y_score_lr,"rf":y_score_rf,"dt":y_score_dt,"knn":y_score_knn, "dummy":y_score_dummy}

# use loop to calculate average precision for all the models
for label in simpleLabels:
    precision[label], recall[label], _ = precision_recall_curve(y_test_plot.ravel(), score[label].ravel())
    average_precision[label] = average_precision_score(y_test_plot, score[label], average="macro")

# setup plot details
models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'kNN', 'Baseline']
colors = ['gold', 'cornflowerblue', 'turquoise', 'darkorange', 'red']
# define the figure size
plt.figure(figsize=(8, 6))

f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []

# loop to init all the model's PR curve
for i in range(5):
    l, = plt.plot(recall[simpleLabels[i]], precision[simpleLabels[i]], color=colors[i], lw=2)
    lines.append(l)
    labels.append(models[i] + ' (area = {0:0.2f})'.format(average_precision[simpleLabels[i]]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.15)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(lines, labels, loc=(0.02, 0.02), prop=dict(size=12))
plt.show()

'''
ROC Curve
'''
# For each class
fpr = dict()
tpr = dict()
roc_auc = dict()
# Plot all ROC curves
plt.figure(figsize=(8, 6))
# Compute micro-average ROC curve and ROC area
for i in range(5):
    fpr[simpleLabels[i]], tpr[simpleLabels[i]], _ = roc_curve(y_test_plot.ravel(), score[simpleLabels[i]].ravel())
    roc_auc[simpleLabels[i]] = auc(fpr[simpleLabels[i]], tpr[simpleLabels[i]])

    plt.plot(fpr[simpleLabels[i]], tpr[simpleLabels[i]],
             label=models[i] + ' (area = {0:0.2f})'.format(roc_auc[simpleLabels[i]]),
             color=colors[i], linestyle=':', linewidth=4)
# plot settings
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


'''
Confusion Matrixs
'''
y_pred_lr = lr.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df_lr = pd.DataFrame(cm_lr,
                     index = ['Easy','Medium','Hard'],
                     columns = ['Easy','Medium','Hard'])
cm_df_knn = pd.DataFrame(cm_knn,
                     index = ['Easy','Medium','Hard'],
                     columns = ['Easy','Medium','Hard'])
cm_df_dt = pd.DataFrame(cm_dt,
                     index = ['Easy','Medium','Hard'],
                     columns = ['Easy','Medium','Hard'])
cm_df_rfc = pd.DataFrame(cm_rfc,
                     index = ['Easy','Medium','Hard'],
                     columns = ['Easy','Medium','Hard'])
#Plotting the confusion matrix
plt.figure(figsize=(18,15))
ax1 = plt.subplot(221)
sns.set(font_scale=2.0)
sns.heatmap(cm_df_lr, annot=True,cmap="Purples", fmt='.20g')
plt.title('Confusion Matrix (LR)')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
ax1 = plt.subplot(222)
sns.set(font_scale=2.0)
sns.heatmap(cm_df_knn, annot=True,cmap="Purples", fmt='.20g')
plt.title('Confusion Matrix (kNN)')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
ax1 = plt.subplot(223)
sns.set(font_scale=2.0)
sns.heatmap(cm_df_dt, annot=True,cmap="Purples", fmt='.20g')
plt.title('Confusion Matrix (DT)')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
ax2 = plt.subplot(224)
sns.set(font_scale=2.0)
sns.heatmap(cm_df_rfc, annot=True,cmap="Purples", fmt='.20g')
plt.title('Confusion Matrix (RF)')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


