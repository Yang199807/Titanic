import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from  sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,plot_roc_curve
from sklearn.svm import SVC

# 1. 数据查看

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print(train.columns)
#print(test.info())


# 2. 数据清洗
#删除 Cabin 列
def preprocessData(df):
    df.Age = df.Age.fillna(df['Age'].quantile())
    df.Fare = df.Fare.fillna(df['Fare'].quantile())
    df.Embarked = df.Embarked.fillna('NAN')
    sexmap = {'male': 0, 'female': 1}
    barkmap = {'C': 0, 'S': 1, "Q": 2, "NAN": 3}
    df['Sex'] = df.Sex.map(sexmap)
    df['Embarked'] = df.Embarked.map(barkmap)
    df['age_bins'] = pd.cut(train['Age'], bins=np.linspace(0, 100, 10),
                               labels=[f"{10 * i}-{10 * i + 10}" for i in range(9)])
    df['fare_bins'] = pd.cut(train['Fare'], bins=np.linspace(0, 100, 10),
                                labels=[f"{10 * i}-{10 * i + 10}" for i in range(9)])
    return df



#print(train.info())
# 3. 数据工程，可视化等

train=preprocessData(train)


# 4. 数据建模分析
x=train.loc[:,['Sex','Age','Pclass','SibSp','Parch','Fare']]
y=train.loc[:,'Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#print("hhh")
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)

# model2=SVC()
# model2.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_score=model.predict_proba(x_test)
result=pd.DataFrame()

# print(accuracy_score(y_test,y_pred))
# print(f"auc:{roc_auc_score(y_test, y_score[:,1])}")
# plot_roc_curve(model, x_test, y_test)
# plt.show()

# y_pred=model2.predict(x_test)
# #y_score=model2.predict_proba(x_test,pr)
# print(accuracy_score(y_test,y_pred))
# plot_roc_curve(model2, x_test, y_test)
#plt.show()
#fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)

#5.预测结果
test=pd.read_csv("test.csv")
test=preprocessData(test)
result=pd.DataFrame()
result['PassengerId']=test.loc[:,"PassengerId"]

X=test.loc[:,['Sex','Age','Pclass','SibSp','Parch','Fare']]
null=X[X.isnull().T.any()]
Y=model.predict(X)

result['Survived']=Y
print(result)
result.to_csv('baseline_submission.csv')