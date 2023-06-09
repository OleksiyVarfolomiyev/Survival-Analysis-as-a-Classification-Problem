# Survival analysis as a classification problem
# Author: Alex (Oleksiy) Varfolomiyev

# an API (i.e., method signatures) for a Python class that will allow the client code to both fit a survival model 
# using the algorithm described in the paper as well as apply the model to unseen data. 

class ClassificationModel:
    def __init__(self, model_name='logistic_regression'):
        self.model_name = model_name
        self.model = self._get_model()

    def _get_model(self):
        if self.model_name == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(fit_intercept=False)
        elif self.model_name == 'random_forest':
            return RandomForestClassifier()
        elif self.model_name == 'svm':
            return SVC()
        else:
            raise ValueError(f"Invalid model_name: {self.model_name}")

    def train(self, X, y):
        self.model.fit(X, y)
        
    def train_test(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_pred, y):
        #y_pred = self.predict(X)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    def evaluate_test(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

#######################################################################
# Algorithm for Survival Data Stacking
#######################################################################
def load_data(n, p):
    import pandas as pd
    import numpy as np

    # Generate random data
    data = np.random.rand(n, p)

    # Create DataFrame
    covariates_df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(1, p+1)])

    import random

    time = [i+1 for i in range(n)]
    status = random.choices([0, 1], k=n)
    covariates_df['time'] = time
    covariates_df['status'] = status
    stack = pd.DataFrame(columns = covariates_df.columns)
    i = 0
    idx = pd.Series()
    while i < n:
        if covariates_df['status'].iloc[i] == 1:
            idx[len(idx)] = i
            stack = pd.concat([stack, covariates_df.iloc[i:n]]) 
        i += 1   
    
    vector = pd.Series()
    for i in np.arange(0,len(idx)):
        if i == 0:
            vector[i] = [1]*(n - idx[i]) + [0]*(len(stack) - (n - idx[i]))
            tmp = n - idx[i]
            
        elif i < len(idx)-1:
            
            vector[i] = [0]*tmp + [1]*(idx[i+1] - idx[i] + 1) 
            tmp = tmp + idx[i+1] - idx[i] + 1
            
            vector[i] = vector[i] + [0]*(len(stack) - len(vector[i]))
        else: 
            vector[i] = [0]*tmp + [1]*(len(stack) - tmp)
            
    for i in np.arange(0, len(idx)):
        stack[i] = vector[i]
    
    stack = stack.drop(['time', 'status'], axis=1)

    y_idx = pd.Series()

    for i in np.arange(0,len(idx)):
        for j in np.arange(0,len(stack)):
            if vector[i][j] == 1:
                y_idx[i] = j
                break

    y = pd.Series([0]*len(stack))
    y[y_idx] = 1

    X = stack
    X.columns = X.columns.astype(str)
    
    X.to_csv('X.csv', index=False)
    y.to_csv('y.csv', index=False)
    
    return X, y