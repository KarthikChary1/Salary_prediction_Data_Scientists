import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/VENUHYMA/Documents/Salary_prediction_Data_Scientists/cleaned_data_2.csv")

#unnecessary columns from the dataset
df_needed=df[['avg_sal','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_competitors','hourly','emp_provided',
             'job_state','same_loc_head','Years_of_estd','python','spark','excel','job_simple','seniority','len_of_desc']]

#get dummies 
df_dummies=pd.get_dummies(df_needed)

#train test split
from sklearn.model_selection import train_test_split
X=df_dummies.drop("avg_sal",axis=1)
y=df_dummies.avg_sal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear regression using ols
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm,X_train,y_train,scoring="neg_mean_absolute_error",cv=3))


# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha=[]
error=[]
for i in range(1,100):
    alpha.append(i/100)
    lm_2=Lasso(alpha=i/100)
    error.append(np.mean(cross_val_score(lm_2,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
import matplotlib.pyplot as plt
plt.plot(alpha,error)

err_check=tuple(zip(alpha,error))
df_error=pd.DataFrame(err_check,columns=(["alpha","error"]))
df_error[df_error.error == max(df_error.error)]

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[200], 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
model=gs.best_estimator_
gs.best_params_

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)


import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

pic=pickle.load(open("model_file.p","rb"))
from data_input import data_in
x1=data_in
