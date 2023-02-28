import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings("ignore")
df=pd.read_csv(r"heart.csv")
#print(df)
#df.info()
#df1=df[df.duplicated()] #used to show how many duplicate values are there#print(df1)
#print(df)
df.drop_duplicates(inplace=True)
#print(df)
#print(df.describe())
"""for i in df.columns:
    print(i,"=>",len(df[i].unique()))"""

#VISUALIZATION
# categorical future analysis with count plot
cat_cols=['sex','exng','caa','cp','fbs','restecg','slp','thall','output']   #categorical features
'''for col in cat_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=col,data=df[cat_cols],hue='output',palette=sns.cubehelix_palette(len(df[col].value_counts())))'''
#plt.show()

#continuing feature analysis
#pair plot
con_columns=["age",'trtbps',"chol",'thalachh','oldpeak','output'] #continuing Features
#sns.pairplot(df[con_columns],hue="output",palette=sns.color_palette(["#000080","#00ffff"]))
#plt.show()
#swarm plot
from sklearn.preprocessing import RobustScaler
df_continuing=df[con_columns]
scaler = RobustScaler()
df_continuing1=scaler.fit_transform(df_continuing.drop(columns="output"))
#print(df_continuing1)
df_dummy=pd.DataFrame(df_continuing1,columns=con_columns[:-1])
#print(df_dummy.head())
df_dummy=pd.concat([df_dummy,df.loc[:,"output"]],axis=1)
#print(df_dummy.head())
df_melt=pd.melt(df_dummy,id_vars="output",var_name="features",value_name="values")
#print(df_melt.head())
#plt.figure(figsize=(8,6))
#sns.swarmplot(x="features",y="values",data=df_melt,hue="output",palette=sns.color_palette(["#2f4f4f","#b22222"]))
#plt.show()
#CORRELATION ANALYSIS WITH HEATMAP
"""plt.figure(figsize=(14,10))
sns.heatmap(df.corr(),annot=True,fmt=".1f")"""
#plt.show()
#FUTURE SELECTION
cat_col=["sex","exng","caa","cp","slp","thall"]
con_col=["age","thalachh","oldpeak"]
#encoding the categorical columns
df1=pd.get_dummies(df,columns=cat_col,drop_first=True)
#print(df1.columns)
x=df1.drop(columns=["output","chol","trtbps","fbs","restecg"])
#print(x.columns)
y=df['output']
#print(df)
#print(x)
#print(y)
# scaling the continuous features
x[con_col]=scaler.fit_transform(x[con_col])
#print(x)
#Train Test Split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=42)
#print("Train_x :",train_x.shape)
#print("Test_x :",test_x.shape)
#print("Train_y :",train_y.shape)
#print("Test_y :",test_y.shape)

# LOGISTIC REGRESSION
from sklearn.model_selection import GridSearchCV
"""from sklearn.linear_model import LogisticRegression
# finding the best parameters
logreg0=LogisticRegression()
grid = {"C": np.logspace(-3,3,7),"penalty":["l1","l2"]}
logreg_cv=GridSearchCV(logreg0,grid,cv=10)
logreg_cv.fit(x,y)
print("best parameters of logistic regression :",logreg_cv.best_params_)
logreg=LogisticRegression(C=logreg_cv.best_params_["C"],penalty=logreg_cv.best_params_["penalty"])
logreg.fit(train_x,train_y)
print("Logistic Regression Accuracy:",logreg.score(test_x,test_y))"""

# KNN
"""from sklearn.neighbors import KNeighborsClassifier

#Finding the best parameters
knn0 = KNeighborsClassifier()
knn_cv = GridSearchCV(knn0, {"n_neighbors": np.arange(1,50)},cv=10)
knn_cv.fit(x,y)
print("Best parameters of kNN :",knn_cv.best_params_)

knn=KNeighborsClassifier(n_neighbors=knn_cv.best_params_["n_neighbors"])
knn.fit(train_x,train_y)
print("KNN Accuracy :",knn.score(test_x,test_y))"""

#SVM

from sklearn.svm import SVC

#finding the best parameters
grid = {"C":np.arange(1,10,1),'gamma':[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]}
svm0 = SVC(random_state=42)
svm_cv = GridSearchCV(svm0,grid,cv=10)
svm_cv.fit(x,y)
#print("Best parameters of SVC:",svm_cv.best_params_)

svm = SVC(C=svm_cv.best_params_["C"],gamma=svm_cv.best_params_["gamma"],random_state=42)
svm.fit(train_x,train_y)
print("SVC Accuracy :",svm.score(test_x,test_y))

#DECISION TREE
'''from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(train_x,train_y)
print("Decision Tree Classifier Accuracy :",tree.score(test_x,test_y))'''

#RANDOM FOREST CLASSIFIER
'''from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_x,train_y)
print("Random Forest Classifier Accuracy :",rf.score(test_x,test_y))

#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(train_x,train_y)
print("Naive Bayes Accuracy :",nb.score(test_x,test_y))

#Cross Validation scores

from sklearn.model_selection import cross_val_score
algorithms = [logreg,knn,tree,rf,svm,nb]

for alg in algorithms:
    accuracies = cross_val_score(estimator=alg,X=x,y=y,cv=10)
    print("{0}: \t {1}".format(alg,accuracies.mean()))

#F1 scores

from sklearn.metrics import f1_score

for alg in algorithms:
    scores = f1_score(test_y,alg.predict(test_x),average=None)
    print("{0}: \t {1}".format(alg,scores))'''

# Classification Reports and Confusion Matrix

#LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report
"""logreg_prediction = logreg.predict(test_x)
#plt.figure()
sns.heatmap(confusion_matrix(test_y,logreg_prediction),annot=True)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Logistic Regression Confusion Matrix")"""
#plt.show()
#print("Logistic Regression Classification Report: \n")
#print(classification_report(test_y,logreg_prediction))

# SVM
svm_prediction = svm.predict(test_x)
#plt.figure()
'''sns.heatmap(confusion_matrix(test_y, svm_prediction),annot=True)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("SVM Confusion Matrix")'''
#plt.show()
print("SVC Classification Report: \n")
print(classification_report(test_y, svm_prediction))
print('asasa',test_x)

# Logistic Regression Curve

"""logreg_pred_proba = logreg.predict_proba(test_x)
#print(logreg_pred_proba)

from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(test_y,logreg_pred_proba[:,1])
plt.figure()
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()"""


pickle.dump(svm,open("model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))






