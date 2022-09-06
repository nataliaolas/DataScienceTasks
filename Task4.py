#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[7]:


dane = pd.read_excel("/Users/Komputer/Downloads/beton.xls","dane2")


# Ex 1

# In[8]:


#wielkosc danych
dane.shape


# In[9]:


#nazwy kolumn
dane.columns


# In[10]:


#wyświetlenie danych z poczatku i konca? 
dane.info()


# In[11]:


# info o każdej kolumnie 
dane.describe()


# In[12]:


#wyswietlanie danych z poczatku 
dane.head()


# In[13]:


#wyswietlanie danych z konca
dane.tail()


# In[14]:


dane.describe()


# In[15]:


dane.nunique()


# In[16]:


dane.isna().sum()


# In[17]:


dane.woda.value_counts()


# In[18]:


dane['wiek_dni'].value_counts()


# In[19]:


dane['wiek_dni'].value_counts().head(1)


# Ex 2

# In[20]:


#X = dane[["3","7","270","180"]]
X = dane.cement 
Y = dane.wytrzymalosc
Z = dane.wiek_dni


# In[21]:


X[dane.wiek_dni]
len(dane.wiek_dni)


# In[25]:


plt.scatter(dane[dane['wiek_dni'] == 3]['cement'],dane[dane['wiek_dni'] == 3]['wytrzymalosc'], color="red", label="3")
plt.scatter(dane[dane['wiek_dni'] == 180]['cement'],dane[dane['wiek_dni'] == 180]['wytrzymalosc'], color="blue", label="180")
plt.scatter(dane[dane['wiek_dni'] == 270]['cement'],dane[dane['wiek_dni'] == 270]['wytrzymalosc'], color="orange", label="270")
plt.scatter(dane[dane['wiek_dni'] == 28]['cement'],dane[dane['wiek_dni'] == 28]['wytrzymalosc'], color="purple", label="28")
plt.scatter(dane[dane['wiek_dni'] == 90]['cement'],dane[dane['wiek_dni'] == 90]['wytrzymalosc'], color="green", label="90")
plt.scatter(dane[dane['wiek_dni'] == 100]['cement'],dane[dane['wiek_dni'] == 100]['wytrzymalosc'], color="tan", label="100")
plt.scatter(dane[dane['wiek_dni'] == 14]['cement'],dane[dane['wiek_dni'] == 14]['wytrzymalosc'], color="coral", label="14")
plt.scatter(dane[dane['wiek_dni'] == 120]['cement'],dane[dane['wiek_dni'] == 120]['wytrzymalosc'], color="deepskyblue", label="120")
plt.ylabel("wytrzymalosc [mPa]")
plt.xlabel("cement[kg/m^3]")
plt.legend()


# In[27]:


import seaborn as sns


# In[28]:


df_dane = dane["wytrzymalosc"].value_counts()
sns.histplot(data=df_dane.index.values)
plt.ylabel("liczba próbek [n]")
plt.xlabel("wytrzymalosc [mPa]")
plt.show()


# Ex 3

# In[27]:


from scipy import stats


# In[28]:


m = dane.kruszywo_grube
n = dane.kruszywo_drobne


# In[29]:


res = stats.linregress(m, n)


# In[30]:


plt.plot(m, n, 'o', label='original data')
plt.plot(m, res.intercept + res.slope*m, 'black', label='fitted line')
plt.legend()
plt.ylabel("Kruszywo drobne", fontsize=12)
plt.xlabel("Kruszywo grube", fontsize=12)
plt.show()


# In[31]:


print("Intercept", res.intercept)
print("Slope", res.slope)
print("Rvalue", res.rvalue)


# Ex 4

# In[32]:


korelacje = abs(dane.corr())
korelacje 


# In[33]:


import seaborn as sns


# In[34]:


dane = sns.heatmap(korelacje,annot=True)


# In[42]:


ax = sns.scatterplot(x='woda', y='suma_waga', data=dane, hue="cement", legend='brief')


# In[43]:


ax = sns.scatterplot(x='kruszywo_grube', y='wiek_dni', data=dane, hue="cement", legend='brief')


# Ex 5 

# In[44]:


slope, intercept, r, p, std_err = stats.linregress(dane.suma_waga,dane.woda)


# In[45]:


def my_regression(x):
    return slope * x + intercept


# In[46]:


m_bet = np.array([2190, 2234, 2290, 2333, 2362, 2412, 2499, 2550])


# In[47]:


szacowana_wartosc = list(map(my_regression, m_bet))
szacowana_wartosc


# In[ ]:




