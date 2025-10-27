import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,LabelEncoder,OrdinalEncoder
df=pd.read_csv("churn-bigml-80.csv")
numerical_Transformer=StandardScaler()
numerical_feature=list(df.select_dtypes(include=["int64","float64"]).columns)
categorical_Transformer=OrdinalEncoder()
categorical_feature=list(df.select_dtypes("object"))
#making the training dataa
encoder=LabelEncoder()
df["Churn"]=encoder.fit_transform(df["Churn"])
x_train=df.drop("Churn",axis=1)
y_train=df["Churn"]
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(transformers=[("num",numerical_Transformer,numerical_feature),("cat",categorical_Transformer,categorical_feature)])                   
##making the pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
pipe=Pipeline([("preprocessor",preprocessor),("classifier",model)])
my_pipe=pipe.fit(x_train,y_train)
# #making the prediction
# df2=pd.read_csv("churn-bigml-20.csv")
# df2["Churn"]=encoder.fit_transform(df2["Churn"])
# x_test=df2.drop("Churn",axis=1)
# y_test=df2["Churn"]
# ##making the prediction
# y_pred=my_pipe.predict(x_test)
# train_pred=my_pipe.predict(x_train)
# #Accuracy
# from sklearn.metrics import accuracy_score
# test_score=accuracy_score(y_pred,y_test)
# train_score=accuracy_score(train_pred,y_train)
# #Saving the model
# import joblib
#joblib.dump(my_pipe,"Churn_prediction.joblib")
# loaded_model=joblib.load("Churn_prediction.joblib")
# x_new=df2.head(10)
# #print(x_new)
# predicition=loaded_model.predict(x_new)
# print(predicition)