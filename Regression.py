import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.svm import SVR
import seaborn as sns
import math
import time
import pandas as pd

class Regression():
    def __init__(self,model_type,x_train,y_train,x_val,y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model_type = model_type
        self.report = pd.DataFrame()
        if self.model_type == "Linear Regression":
            self.model = LinearRegression()
        elif self.model_type == 'Decision Tree Regression':
            self.model = DecisionTreeRegressor(random_state=42)
        elif self.model_type == 'Random Forest Regression':
            self.model = RandomForestRegressor()
        elif self.model_type == 'SVM Regression':
          self.model = SVR()
   
    def result(self):
       mse_t = mean_squared_error(self.y_train, self.y_true)
       mse_v = mean_squared_error(self.y_val, self.y_validated)
       mse = [mse_t,mse_v]
       rmse = [math.sqrt(mse_t),math.sqrt(mse_v)]
       df = pd.DataFrame({'MSE':mse,'RMSE':rmse},index=['train','validation'])
       return df
    def get_feature_importances(self):
 
        if (self.model_type == 'Decision Tree Regression') or (self.model_type == 'Random Forest Regression') or (self.model_type == 'SVM Regression'):    
            self.feature_importances_table = pd.DataFrame(self.best_model.feature_importances_,
                                                    index = self.x_train.columns,
                                                    columns=['Importance']).sort_values('Importance',ascending =False)
            plt.figure(figsize=(9,7.5))
            self.feature_importances_bar = sns.barplot(y= self.feature_importances_table.index[:15], x= self.feature_importances_table['Importance'][:15])
            plt.show()
            return self.feature_importances_bar  
        else:
            return print('This classification method does not have the attribute feature importance.')  
    def scores(self,best_model,value):
        self.score_table = pd.DataFrame()
        self.score_train = self.best_model.score(self.x_train,self.y_train)
        self.score_val = self.best_model.score(self.x_val,self.y_val)
        d = {'Model Name': [self.model_type],
             'Train Score': [self.score_train], 
             'Validation Score': [self.score_val],
             'Score Difference':[self.score_train-self.score_val]}
        self.scores_table = pd.DataFrame(data=d)
        return self.scores_table
    def show_kde(self):
        sns.kdeplot(self.y_val, self.y_validated, 
            color='r', shade=True, Label='Iris_Setosa', 
            cmap="Reds", shade_lowest=False) 
        
    def show(self):
        fig, ax = plt.subplots(1,1)
        labels = [] 
        for s in self.df:
            self.df[s].plot(kind='density',title = 'Distribuation of Actual value and Prediction Value',label=s)
            labels.append(s)
        plt.legend(labels)

        fig.show()

 
    def get_scores(self,params,cv_type):
        start = time.time()
        model = self.model
        opt_model = GridSearchCV(model,
                                 params,
                                 cv=cv_type,
                                 scoring='r2',
                                 return_train_score=True,
                                 n_jobs=-1)
        
        self.opt_model = opt_model.fit(self.x_train,self.y_train) 
       
        self.best_model = opt_model.best_estimator_
        value = opt_model.best_score_
        self.scores = self.scores(self.best_model,value)
        self.best_params = opt_model.best_params_
        display(self.scores_table)
        print()
        if params == {}:
            pass
        else:
            print("The best hyperparameters are: ", self.best_params,'\n')
        self.y_validated = self.best_model.predict(self.x_val)

        self.y_true = self.best_model.predict(self.x_train)
        self.report = self.result()
        stop = time.time()
        print(f"time taken by hayper perameter for searching best perameter : {stop - start} s \n",)
        df = pd.DataFrame({'Actual':self.y_val,'Prediction':self.y_validated})
        self.df = df
        display(df[:5])
        return self.report


     

 