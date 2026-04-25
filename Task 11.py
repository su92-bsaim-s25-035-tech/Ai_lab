import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

data= pd.read_csv('Task 11/amazon.csv')
frequent_val= data['rating'].mode()[0]
data['rating']= data['rating'].fillna(frequent_val)

text_cols= data.select_dtypes(include=['object', 'string']).columns
for col in text_cols:
    data[col]= pd.factorize(data[col])[0]

x= data.iloc[:, 0: -1]
y= data.iloc[:, -1]

X_train, X_test, Y_train, Y_test= train_test_split(x,y, test_size=0.3, shuffle=True, random_state=42)
print("---Training Models...Please wait---")

Dtree= DecisionTreeClassifier()
Dtree.fit(X_train, Y_train)
Y_dpred= Dtree.predict(X_test)
d_acc= metrics.accuracy_score(Y_test, Y_dpred)

RF= RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
RF.fit(X_train,Y_train)
Y_rpred= RF.predict(X_test)
r_acc= metrics.accuracy_score(Y_test, Y_rpred)

GausNB= GaussianNB()
GausNB.fit(X_train, Y_train)
Y_gpred= GausNB.predict(X_test)
g_acc= metrics.accuracy_score(Y_test, Y_gpred)

print("\n---Accuracy Scores---")
print(f"Decision Tree Accuracy: {d_acc:.4f}")
print(f"Random Forest Accuracy: {r_acc:.4f}")
print(f"GaussianNB Accuracy: {g_acc:.4f}")

plt.figure(figsize=(10,6))
model_names= ['Decision Tree', 'Random Forest', 'Gaussian']
accuracy_vals= [d_acc, r_acc, g_acc]

plt.bar(model_names, accuracy_vals, color=['blue', 'green', 'orange'])
plt.title("Comparison of Classifiers accuracy")
plt.xlabel("Classifiers")
plt.ylabel("Accuracy Scores")
plt.ylim(0,1)
plt.show()
