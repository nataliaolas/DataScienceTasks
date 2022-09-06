#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model
from sklearn.metrics import r2_score


# In[8]:


dane = pd.read_csv("/Users/Komputer/Downloads/dane.csv")


# Ex 1:

# In[9]:


from numpy.polynomial import Polynomial


# In[10]:


wx = dane.Argumenty
wy = dane.Wartosci


# In[11]:


st = 6 
poly = Polynomial.fit( wx,wy, st)
print( poly )
print( poly.coef )
print("================================")


# In[16]:


plt.grid()
plt.scatter(wx,wy)
model_p = poly;
x = np.arange(0, 12, 0.2)
y = model_p(x)
plt.plot(x,y,"-k")
plt.show()


# In[15]:


print(r2_score(dane.Wartosci,model_p(dane.Argumenty)))


# In[17]:


print(model_p(0.6))
print(model_p(3.8))
print(model_p(10.0))


# Ex 2:

# In[18]:


test = dane.sample(frac=0.2, replace=False, random_state=1)
train = dane[~dane.Wartosci.isin(test.Wartosci)]


# In[19]:


test


# In[20]:


train


# In[21]:


st = 6 
poly = Polynomial.fit( train.Argumenty,train.Wartosci, st)
print( poly )
print( poly.coef )
print("================================")


# In[22]:


plt.grid()
plt.scatter(train.Argumenty,train.Wartosci)


# In[23]:


print(r2_score(train.Wartosci,model_p(train.Argumenty)))


# In[24]:


plt.grid()
plt.scatter(train.Argumenty,train.Wartosci)
model_p = poly;
## liczby od 0 do 9 z krokiem 0.2
x = np.arange(-1, 12, 0.2)
y = model_p(x)
plt.plot(x,y,"-k")
plt.show()


# In[25]:


print(r2_score(test.Wartosci,model_p(test.Argumenty)))


# In[26]:


plt.grid()
plt.scatter(test.Argumenty,test.Wartosci)
model_p = poly;
x = np.arange(-1, 12, 0.2)
y = model_p(x)
plt.plot(x,y,"-k")
plt.show()


# In[28]:


plt.grid()
plt.scatter(train.Argumenty, train.Wartosci, c='purple')
plt.scatter(test.Argumenty, test.Wartosci, c='yellow')
model_p = poly;
x = np.arange(-1, 12, 0.2)
y = model_p(x)
plt.plot(x,y,"-k")


# Ex 3:

# In[29]:


from sklearn import linear_model
import itertools as it


# In[31]:


beton = pd.read_excel("/Users/Komputer/Downloads/beton.xls")


# In[32]:


plt.grid()
plt.scatter(beton.cement, beton.wytrzymalosc, c='green')


# In[34]:


cement = beton.cement
cement = cement.to_numpy()
cement = cement[:, None]
regr = linear_model.LinearRegression()
regr.fit(cement, beton.wytrzymalosc)
print(f"wspolczynnik regresji dla cement i wytrzymalosci: {regr.coef_}")
print(f"R^2 dla wytrzymalosci i cementu:  {regr.score(cement, beton.wytrzymalosc)}")


# In[36]:


X = beton[['cement','woda', 'wiek_dni']]
y = beton['wytrzymalosc']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(f"wspolczynnik regresji dla cement-woda-wiek_dni i wytrzymalosci: {regr.coef_}")
print(f"R^2 dla cement-woda-wiek_dni i wytrzymalosci:  {regr.score(X, y)}")


# Ex 4:

# In[37]:


def get_best_combination_and_r2_score(list_of_combinations):
    y = beton.wytrzymalosc
    max_r2_score = -1
    best_match = []
    regr = linear_model.LinearRegression()
    for combination in list_of_combinations:
        data = pd.DataFrame()
        for x in combination:
            data[x] = beton[x]
        X = pd.DataFrame(data=data, columns=list(combination))
        regr.fit(X, y)
        if regr.score(X,y) > max_r2_score:
            max_r2_score = regr.score(X,y)
            best_match = combination
    return max_r2_score, best_match


# In[38]:


columns = beton.columns
columns = columns.drop('wytrzymalosc')


# In[40]:


max_r2_score, best_combination = get_best_combination_and_r2_score(list(it.combinations(columns, 3)))
print(f"Najwyzsza wartosc R^2 miały: {best_combination}")
print(f"Najwyzsza wartosc R^2 miały: {max_r2_score}")


# In[41]:


X = beton.drop(columns='wytrzymalosc')
y = beton.wytrzymalosc


# In[43]:


regr = linear_model.LinearRegression()
regr.fit(X, y)
print(f"wspolczynnik regresji dla wszystkich zmiennych niezaleznych i wytrzymalosci: {regr.coef_}")
print(f"R^2 dla wszystkich zmiennych niezaleznych i wytrzymalosci:  {regr.score(X, y)}")
r2_score_for_all_columns = regr.score(X,y)


# Ex 5:

# In[45]:


max_r2_score, best_combination = get_best_combination_and_r2_score(list(it.combinations(columns, 5)))
print(f"Najwyzsza wartosc R^2 dla 5 kolumn: {best_combination}")
print(f"Najwyzsza wartosc R^2 dla 5 kolumn: {max_r2_score}")


# In[47]:


diff_proc1 = (1 - max_r2_score/r2_score_for_all_columns) *100
print(f"Roznica procentowa R^2 dla 8 atrybutów wynosi: {diff_proc1}")


# In[ ]:




