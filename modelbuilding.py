import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('D:/ds project/eda_performeddata.csv')

# choose relevant columns
print(df.columns)

df_model=df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided','job_state','same_state','age','python_yn','aws_yn','spark_yn','excel_yn','job_simp','seniority','desc_len']]

# get dummy variables
df_dummy = pd.get_dummies(df_model, dtype=int)

# train test split
from sklearn.model_selection import train_test_split

X=df_dummy.drop('avg_salary',axis=1)

y=df_dummy.avg_salary.values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# multiple linear regression
#1] ols regression
import statsmodels.api as sm
X = X.astype(float)
y = y.astype(float)


X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
print(model.fit().summary())

#2] from sklearn, cross validation 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

print(np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

# lasso regression
lm_l=Lasso(alpha=0.17)
lm_l.fit(X_train,y_train)
print(np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

alpha=[]
error=[]
for i in range(1,100):
    alpha.append(i/100)
    lml=Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
plt.plot(alpha,error)
plt.show()

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
print(df_err[df_err.error == max(df_err.error)])

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(random_state=42,
    n_jobs=-1)
print(np.mean(cross_val_score(rf,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)))

# tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
parameters={'n_estimators':range(10,300,10),'criterion':('squared_error', 'absolute_error'),'max_features':('sqrt', 'log2')}
gs=GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3, n_jobs=-1)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_estimator_)

# test ensembles
tpred_lm=lm.predict(X_test)
tpred_lml=lm_l.predict(X_test)
tpred_rf=gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,tpred_lm))
print(mean_absolute_error(y_test,tpred_lml))
print(mean_absolute_error(y_test,tpred_rf))

print(mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2))


import pickle
pickl={'model':gs.best_estimator_}
pickle.dump(pickl,open('model_file'+'.p','wb'))

file_name='model_file.p'
with open (file_name,'rb') as pickled :
    data=pickle.load(pickled)
    model=data['model']
    
print(model.predict(X_test.iloc[1,:].values.reshape(1,-1)))
print(list(X_test.iloc[1,:]))