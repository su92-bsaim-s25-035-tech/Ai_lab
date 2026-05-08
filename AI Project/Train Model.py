import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")

#Reading Csv
df= pd.read_csv("dress_code_dataset.csv")
df.head()
df.tail()

#number of rows and colums
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
#null values
df.isnull().sum()
#datatypes
df.dtypes
#checking null values
print("\nNull values before clening:")
print(df.isnull().sum())

#filling null values
num =df['shirt_color'].mode()[0]
df['shirt_color']=df["shirt_color"].fillna(num)
num=df['shoe_type'].mode()[0]
df['shoe_type'].fillna(num)
num=df['has_id_card'].mode()[0]
df['has_id_card'].fillna(num)
print("\nNull values after cleanig:")
print(df.isnull().sum())

#unique values
df['department'].unique()
df['shirt_type'].unique()

#droping unimportant columns
df.drop('employee_id', axis=1, inplace=True)
df.drop('name', axis=1, inplace=True)

df.dtypes

#splitting into x and y
x= df.iloc[:,0:-1]
x.shape
y=df.iloc[:, -1]
y.shape

#object to int columns
cat_columns= x.select_dtypes(['object']).columns
x[cat_columns]= x [cat_columns].apply(lambda x: pd.factorize(x)[0])

#train test splitting
X_train, X_test, Y_train, Y_test= train_test_split(x,y,test_size=.3,shuffle=False)

#Applying classifires
#random forest
RF= RandomForestClassifier()
RF.fit(X_train, Y_train)
Y_rpred = RF.predict(X_test)
r_accuracy = accuracy_score(Y_test, Y_rpred)
print('Accuracy: %f' % r_accuracy)
r_precision = metrics.precision_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print('Precision: %f' % r_precision)
r_recall = metrics.recall_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print('Recall: %f' % r_recall)
r_f1 = metrics.f1_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print('F1 score: %f' % r_f1)

#decision tree
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train,Y_train)
Y_dpred = Dtree.predict(X_test)
d_accuracy = accuracy_score(Y_test,Y_dpred)
print('Accuracy: %f'% d_accuracy)
d_precision = metrics.precision_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print('Precision: %f' % d_precision)
d_recall = metrics.recall_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print('Recall: %f' % d_recall)
d_f1 = metrics.f1_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print('F1 score: %f' % d_f1)

#KNN
KNN = KNeighborsClassifier()
KNN.fit(X_train,Y_train)
Y_kpred = KNN.predict(X_test)
k_accuracy = accuracy_score(Y_test,Y_kpred)
print('Accuracy: %f' % k_accuracy)
k_precision = metrics.precision_score(Y_test, Y_kpred, average='weighted', labels=np.unique(Y_kpred))
print('Precision: %f' % k_precision)
k_recall = metrics.recall_score(Y_test, Y_kpred, average='weighted', labels=np.unique(Y_kpred))
print('Recall: %f' % k_recall)
k_f1 = metrics.f1_score(Y_test, Y_kpred, average='weighted', labels=np.unique(Y_kpred))
print('F1 score: %f' % k_f1)

#line graph
plt.figure(figsize=(16,10))
x1 = np.array(['Random Forest', 'Decision Tree', 'KNeighbors'])
y1 = np.array([r_accuracy, d_accuracy, k_accuracy])
plt.plot(x1, y1, marker='o', label='Accuracy')
x1 = np.array(['Random Forest', 'Decision Tree', 'KNeighbors'])
y1 = np.array([r_precision, d_precision, k_precision])
plt.plot(x1, y1, marker='o', label='Precision')
x1 = np.array(['Random Forest', 'Decision Tree', 'KNeighbors'])
y1 = np.array([r_recall, d_recall, k_recall])
plt.plot(x1, y1, marker='o', label='Recall')
x1 = np.array(['Random Forest', 'Decision Tree', 'KNeighbors'])
y1 = np.array([r_f1, d_f1, k_f1])
plt.plot(x1, y1, marker='o', label='F1')
plt.title("Scores of Applied Classifiers")
plt.legend()
plt.savefig('static/line_graph.png')
plt.show()

#bar graph
left = [1, 2, 3]
plt.figure(figsize=(16, 10))
height = [r_f1, d_f1, k_f1]
tick_label = ['Random Forest', 'Decision Tree', 'KNeighbors']
plt.bar(left, height, tick_label=tick_label, width=0.9, color=['#08737f', '#089f8f', '#64c987'])
plt.xlabel('Classifiers')
plt.ylabel('F1 Scores')
plt.title('F1 Scores of Applied Classifiers')
plt.savefig('static/bar_graph.png')
plt.show()

with open('models/dress_code_model.pkl', 'wb') as f:
    pickle.dump(RF, f)
 
factorize_mappings = {}
dataset_original = pd.read_csv('dress_code_dataset.csv')
dataset_original['shirt_color'] = dataset_original['shirt_color'].fillna(dataset_original['shirt_color'].mode()[0])
dataset_original['shoe_type'] = dataset_original['shoe_type'].fillna(dataset_original['shoe_type'].mode()[0])
dataset_original['has_id_card'] = dataset_original['has_id_card'].fillna(dataset_original['has_id_card'].mode()[0])
dataset_original.drop(['employee_id', 'name'], axis=1, inplace=True)
 
for col in cat_columns:
    codes, uniques = pd.factorize(dataset_original[col])
    factorize_mappings[col] = {val: idx for idx, val in enumerate(uniques)}
 
with open('models/factorize_mappings.pkl', 'wb') as f:
    pickle.dump(factorize_mappings, f)
 
print("\nModel and mappings saved successfully")



 




