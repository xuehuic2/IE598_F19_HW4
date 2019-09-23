

import pandas as pd

from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('housing2.csv')
df=df.dropna()

df.shape
df.head()

df.describe()

#Describe the data sufficiently using the methods and visualizations that we used previously in Module 3 
#and again this week.  Include any output, graphs, tables, heatmaps, box plots, etc. 
#Label your figures and axes. DO NOT INCLUDE CODE!


# In[162]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[163]:


sns.set(style='whitegrid', context='notebook')
#describe some of the important features
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

colss= ['ATT1','ATT2','ATT3','ATT4','ATT5','ATT6','ATT7','ATT8','ATT9','ATT10','ATT11','ATT12','ATT13','CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[196]:


sns.pairplot(df[cols], height=2)
plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()


# In[165]:


#see the boxplot of CHAS with MEDV. Are there any significant difference of 
#Median value of owner-occupied homes with Charles River dummy variable, 1 if tract bounds 
#river; 0 otherwise
_ = sns.boxplot(x='CHAS', y='MEDV', data=df)

# Label the axes
_ = plt.xlabel('CHAS')

_ = plt.ylabel('MEDV')
# Show the plot
plt.show()


# In[209]:



import numpy as np

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()


# In[167]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[168]:


#Split data into training and test sets.  
X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[169]:


#Part 2: Linear regression
#Fit a linear model using SKlearn to all of the features of the dataset.  
#Describe the model (coefficients and y intercept), plot the residual errors, calculate performance metrics: MSE and R2.  


# In[170]:


import statsmodels.api as sm
X = df.iloc[:, :-1].values
y = df['MEDV'].values
#X = sm.add_constant(X)


# In[171]:


model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()
#using ordinary least squares model


# In[172]:


#estimate the coefficient and intercept of the SKlearn 
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)#using all the data to fit model
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# In[173]:


##R² score of our model,this is the percentage of explained variance of the predictions. 
slr.score(X,y)


# In[174]:


colss= ['ATT1','ATT2','ATT3','ATT4','ATT5','ATT6','ATT7','ATT8','ATT9','ATT10','ATT11','ATT12','ATT13','CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


# In[175]:


#get the coefficient of each features
df_1 = pd.DataFrame(slr.coef_)
df_2 = pd.DataFrame(colss)
coefficient= pd.concat([df_2, df_1],axis=1, ignore_index=True)
coefficient


# In[176]:


#evaluate the performance of the linear regression model using predictions and test set
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[177]:


#plot the residual errors
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('./figures/slr_residuals.png', dpi=300)
plt.show()


# In[178]:


#calculate performance metrics: MSE and R2.  
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[179]:


#Part 3.1: Ridge regression
#Fit a Ridge model using SKlearn to all of the features of the dataset.
#Test some settings for alpha.  
#Describe the model (coefficients and y intercept),
#plot the residual errors, calculate performance metrics: MSE and R2.  
#Which alpha gives the best performing model?


# In[210]:


from sklearn.linear_model import Ridge


# In[211]:


#k_range = np.arange(0.0, 1.0, 0.1)
k_range = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
for k in k_range:
    ridge = Ridge(alpha=k)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    print('alpha=',k,'MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('alpha=',k ,'R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
#we choose alpha as 1.0


# In[212]:


#coef_
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
#print(ridge.coef_)
df_4 = pd.DataFrame(ridge.coef_)
df_2 = pd.DataFrame(colss)
coefficient3= pd.concat([df_2, df_4],axis=1, ignore_index=True)
coefficient3


# In[186]:


#intercept
print('Intercept: %.3f' % ridge.intercept_)


# In[213]:


#plot the residual errors
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('./figures/slr_residuals.png', dpi=300)
plt.show()


# In[188]:


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[189]:


#Part 3.2: LASSO regression
#Fit a LASSO model using SKlearn to all of the features of the dataset.  
#Test some settings for alpha.  
#Describe the model (coefficients and y intercept), 
#plot the residual errors, calculate performance metrics: MSE and R2.  
#Which alpha gives the best performing model?


# In[190]:


from sklearn.linear_model import Lasso


# In[191]:


#k_range = np.arange(0.0, 1.0, 0.1)
k_range = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
for k in k_range:
    lasso = Lasso(alpha=k)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print('alpha=',k,'MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('alpha=',k ,'R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

#here we get while alpha=0.4, we have a small MSE and large R^2


# In[192]:


lasso = Lasso(alpha=0.4)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
#print(lasso.coef_)
df_3 = pd.DataFrame(lasso.coef_)
df_2 = pd.DataFrame(colss)
coefficient3= pd.concat([df_2, df_3],axis=1, ignore_index=True)
coefficient3


# In[193]:


#intercept
print('Intercept: %.3f' % lasso.intercept_)


# In[194]:


#plot the residual errors
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('./figures/slr_residuals.png', dpi=300)
plt.show()


# In[195]:


#calculate performance metrics

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[ ]:


print("My name is {Xuehui Chao}")
print("My NetID is: {xuehuic2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

