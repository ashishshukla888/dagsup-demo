import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

import dagshub
dagshub.init(repo_owner='ashishshukla888', repo_name='dagsup-demo', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/ashishshukla888/dagsup-demo.mlflow")

#load data
iris = load_iris()
X = iris.data 
y = iris.target

#split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#define params
max_depth = 15
n_estimators = 10

#apply mlflow
mlflow.set_experiment('iris_dt')

with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    
    
    # create a cm
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save the plot
    plt.savefig("cm.png")
    #mlflow code log
    mlflow.log_artifact("cm.png")
    # model log
    mlflow.sklearn.log_model(dt,"decesion tree")


    mlflow.log_artifact(__file__)

    #tag
    mlflow.set_tag('author','asd')
    mlflow.set_tag('model','dt')

    print('accuracy',accuracy)


