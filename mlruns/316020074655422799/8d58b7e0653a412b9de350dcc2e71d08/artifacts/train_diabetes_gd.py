import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow

df = pd.read_csv('diabetes.csv')
print(df.isnull().sum())

x = df.drop('Outcome',axis=1)
y= df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

rf = RandomForestClassifier()

param_grid = {
    'n_estimators': [10,50,100],
    'max_depth': [None,10,20,30]

}

#Applying grid search cv
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)

mlflow.set_experiment('daibetes-rf')

with mlflow.start_run() as parent:

    grid_search.fit(x_train,y_train)

    #log all the children
    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy',grid_search.cv_results_['mean_test_score'][i])

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #params
    mlflow.log_params(best_params)

    #metrics
    mlflow.log_metric("accuracy",best_score)

    #data
    train_df = x_train
    train_df['Outcome'] = y_train
    train_df= mlflow.data.from_pandas(train_df)

    mlflow.log_input(train_df,"training")

    test_df = x_test
    test_df['Outcome'] = y_test
    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df,"validation")

    # source code
    mlflow.log_artifact(__file__)

    #model
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random forest")

    #tags
    mlflow.set_tag('author','ayush')

    print(best_params)
    print(best_score)
