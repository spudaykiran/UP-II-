import pandas as pd
import numpy as np
import pickle
df=pd.read_csv("Heart.csv")
df["Sex"]=df["Sex"].replace({'F':0,'M':1})
df["ChestPainType"]=df["ChestPainType"].replace({'ATA':0,'NAP':1,'TA':2,'ASY':3})
df["RestingECG"]=df["RestingECG"].replace({'Normal':0,"ST":1,"LVH":2})
df["ExerciseAngina"]=df["ExerciseAngina"].replace({"N":0,"Y":1})
df["ST_Slope"]=df["ST_Slope"].replace({"Up":0,"Flat":1,"Down":2})
X=df.drop("HeartDisease",axis=1)
y=df["HeartDisease"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
RF=RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=5)
RF.fit(X_train,y_train)
pickle.dump(RF, open("Model.pkl", "wb"))