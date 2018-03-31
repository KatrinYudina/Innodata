
# coding: utf-8

# ### Задание:
# 
#     Имеются данные продаж домов. 
#     Нужно реализовать модель Лассо регрессии, используя координатный спуск, либо любую другую модель регрессии, 
#     на ваше усмотрение. 
#     Если вы считаете, что она лучше (выбор объяснить и представить сравнение результатов). 
#     При этом, для выполнения работы не разрешается использовать библиотеки с реализованными алгоритмами регрессии, 
#     но для операций над матрицами можно использоватья numpy (или аналогичные библиотеки для Scala).
# 
#     Представить в виде нескольких слайдов этапы работы и полученные результаты. 
#     Желательно использовать Python 3.6 , либо Scala 2.12. Чтобы мы могли проверить ваши результаты прикрепите, 
#     пожалуйста, список библиотек (pip freeze > requirements.txt) для Python, либо build.sbt для Scala.

# In[170]:


import numpy as np

import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


# # Что написать про weight и куда???

# In[236]:


class CoordinateDescentLasso:
    """
    This is implementation of coordinate descent algorithm 
    to solve the LASSO problem
    """
    
    def __init__(self, iter_, lambda_, lr, fit_intercept=True):
        
        """
        Args:
        
            iter_: int
            Default is set as 1000, the same as sklearn.
        
            lambda_: float
            Regularization parameter.
        
            lr: float
            Learning rate
        
            fit_intercept: boolean
            Intercept is the value at which the fitted line crosses the y-axis.
            If it is True we using this meaning for prediction.
        """
        self.lambda_ = lambda_
        self.weights = None
        self.iter = iter_
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.intercept = None
        
    def lasso(self, weight, x, y):
        """
        Derivative of Lasso regression.
        """
        for iter_ in range(50): # number itterations of gradient descent
            mlt = np.multiply((weight * x - y), x)
            dL_dw = 2 * np.mean(mlt)/x.shape[0] + self.lambda_ * np.sign(weight)
            weight = weight - self.lr * dL_dw
        return weight
    
    def loss(self, x, y):
        """
        Loss function.
        """
        diff = np.dot(x, np.expand_dims(self.weights.T, axis=1)) - np.expand_dims(y, axis=1)
        return np.sum(np.square(diff)) + self.lambda_ * np.sum(np.abs(self.weights))
    
    def fit(self, x, y, show_iters = False):
        """
        Fiting our model.
        
        Args:
            x, y: arrays
            X_train and y_train data set
            
            show_iter: boolean
            If it is True, during implementation every 100 iterations will be shown loss meaning.  
        """
        self.weights = np.random.normal(size=x.shape[1])
        for iter_ in tqdm(range(self.iter)):
            if show_iters:
                if iter_ % 100 == 0:
                    print("iter: {}, loss: {}".format(iter_, self.loss(x, y)))
            for feature_iter in range(x.shape[1]):
                self.weights[feature_iter] = self.lasso(self.weights[feature_iter], x[:, feature_iter], y)
                
                if self.fit_intercept:
                    self.intercept = np.sum(y - np.dot(x, self.weights))/(x.shape[0])
            
    def predict(self, x):
        """
        Prediction after using coordinate descent.
        """
        y = np.dot(x, self.weights)
        
        if self.fit_intercept:
            y += self.intercept
        return y
        


# In[79]:


df = pd.read_csv("history_sales.csv")


# In[9]:


df.head()


# In[70]:


y = df['price']
X = df.drop(['price', 'date'], axis = 1)


# In[71]:


y_train, y_test, X_train, X_test = train_test_split(y, X, test_size = 0.3)


# In[237]:


regr = CoordinateDescentLasso(iter_ = 1000, lambda_= 20, lr=2)


# In[238]:


regr.fit(normalize(X_train.as_matrix()), y_train.as_matrix())


# In[239]:


regr.predict(normalize(X_test.as_matrix()))


# In[240]:


np.sqrt(mean_squared_error(regr.predict(normalize(X_test.as_matrix())), y_test))

