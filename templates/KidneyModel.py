import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("Kidney.csv",encoding="utf-8")
X=df.drop("Chronic Kidney Disease",axis=1)
y=df["Chronic Kidney Disease"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,y_train.shape,X_test.shape,y_test.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score

RF=RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=5)
RF.fit(X_train,y_train)
pickle.dump(RF, open("KidneyModel.pkl", "wb"))