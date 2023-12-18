import pandas as pd
import numpy as np
import pickle
df=pd.read_csv("Liver.csv",encoding="utf-8")
df.dropna(inplace=True)
df["Gender of the patient"]=df["Gender of the patient"].replace({"Male":1,"Female":0})
df["Result"]=df["Result"].replace({2:0})
X=df.drop("Result",axis=1)
y=df["Result"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
KNN=KNeighborsClassifier(n_neighbors=7)
KNN.fit(X_train,y_train)
pickle.dump(KNN, open("LungModel.pkl", "wb"))