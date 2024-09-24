from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import numpy as np

df = pd.read_csv('diabetes.csv')
# Replace zero values with NaN in columns where zero is not a valid value
cols_with_missing_vals = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing_vals] = df[cols_with_missing_vals].replace(0, np.nan)

# Impute the missing values with the mean of the respective column
df.fillna(df.mean(), inplace=True)


# Split into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Optional: Scale the data for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Define the objective function
def objective(trial):
    # Suggest values for the hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    # Create the RandomForestClassifier with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # Perform 3-fold cross-validation and calculate accuracy
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

    return score  # Return the accuracy score for Optuna to maximize

study = optuna.create_study(direction='maximize')  # We aim to maximize accuracy
study.optimize(objective, n_trials=50)  # Run 50 trials to find the best hyperparameters



mlflow.set_experiment('daibetes-rf-hp')

with mlflow.start_run():

   

    # Train a RandomForestClassifier using the best hyperparameters from Optuna
    best_model = RandomForestClassifier(**study.best_trial.params, random_state=42)

    # Fit the model to the training data
    best_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred)

    # Print the test accuracy
    print(f'Test Accuracy with best hyperparameters: {test_accuracy:.2f}')

   

    
    #params
    mlflow.log_params(study.best_trial.params)

    #metrics
    mlflow.log_metric("accuracy",test_accuracy)

    #data
    X_train_df = pd.DataFrame(X_train, columns=df.drop('Outcome', axis=1).columns)
    X_test_df = pd.DataFrame(X_test, columns=df.drop('Outcome', axis=1).columns)

    # Add the 'Outcome' column back
    X_train_df['Outcome'] = y_train.values
    X_test_df['Outcome'] = y_test.values

    # Log input data with MLflow
    mlflow.log_input(mlflow.data.from_pandas(X_train_df), "training")
    mlflow.log_input(mlflow.data.from_pandas(X_test_df), "validation")

    # source code
    mlflow.log_artifact(__file__)

    #model
    mlflow.sklearn.log_model(best_model, "random_forest")

    #tags
    mlflow.set_tag('author','ayush')

