import pandas as pd
import numpy as np
import pickle
df=pd.read_csv("Diabetes.csv",encoding="utf-8")
df["gender"]=df["gender"].replace({"Male":1,"Female":0,"Other":2})
df["smoking_history"]=df["smoking_history"].replace({"never":0,"No Info":1,"current":2,"former":3,"ever":4,"not current":5})
X=df.drop("diabetes",axis=1)
y=df["diabetes"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
GB=GradientBoostingClassifier(n_estimators=100)
GB.fit(X_train,y_train)
pickle.dump(GB, open("Model.pkl", "wb"))