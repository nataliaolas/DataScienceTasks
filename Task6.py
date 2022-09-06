#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Zadanie 1 

# In[3]:


dane = pd.read_csv("/Users/Komputer/Downloads/haberman.csv")


# In[4]:


dane.head()


# In[15]:


wiek_przezytych = dane[dane.status_przezycia==1]["wiek"]
wiek_umarlych = dane[dane.status_przezycia==2]["wiek"]
rok_przezytych = dane[dane.status_przezycia==1]["rok"]
rok_umarlych = dane[dane.status_przezycia==2]["rok"]
line1 = plt.scatter(wiek_przezytych,rok_przezytych,marker=",",c="blue", s=12)
line2 = plt.scatter(wiek_umarlych,rok_umarlych,marker=".",c="green", s=15)
plt.grid(color='grey',linestyle="-",linewidth="0.4")
plt.xlabel("wiek pacjenta w czasie operacji[n]")
plt.ylabel("rok w którym była operacja")
plt.legend(['powyżej 5 lat', 'poniżej 5 lat'])
plt.show()


# Zadanie 2

# In[6]:


from sklearn.neighbors import KNeighborsClassifier


# In[7]:


X = dane[["wiek","rok"]]
y = dane.status_przezycia
K = 1


# In[8]:


nbrs = KNeighborsClassifier(n_neighbors=K, algorithm='ball_tree',weights='distance').fit(X,y)


# In[9]:


print(nbrs.score(X,y))


# In[10]:


distances, indices = nbrs.kneighbors(X)
print(distances)


# In[11]:


data60_65 = {'wiek': [40,55], 'rok':[60,65]}
age = pd.DataFrame(data=data60_65)
print(nbrs.predict(age))


# In[12]:


X = dane[["wiek","rok"]]
y = dane.status_przezycia
K = 3
nbrs = KNeighborsClassifier(n_neighbors=K,weights='distance').fit(X,y)
print(nbrs.score(X,y))


# In[13]:


distances, indices = nbrs.kneighbors(X)
print(distances)


# In[74]:


data60_65 = {'wiek': [40,55], 'rok':[60,65]}
age = pd.DataFrame(data=data60_65)
print(nbrs.predict(age))
distances, indices = nbrs.kneighbors(X)
print(distances)


# In[75]:


X = dane[["wiek","rok"]]
y = dane.status_przezycia
K = 2
nbrs = KNeighborsClassifier(n_neighbors=K,weights='distance').fit(X,y)
print(nbrs.score(X,y))
distances, indices = nbrs.kneighbors(X)
print(distances)


# In[77]:


data60_65 = {'wiek': [40,55], 'rok':[60,65]}
age = pd.DataFrame(data=data60_65)
print(nbrs.predict(age))


# In[67]:


X = dane[["wiek","rok"]]
y = dane.status_przezycia
K = 1
nbrs = KNeighborsClassifier(n_neighbors=K, algorithm='ball_tree',weights='uniform').fit(X,y)
print(nbrs.score(X,y))


# In[68]:


distances, indices = nbrs.kneighbors(X)
print(distances)


# In[69]:


data60_65 = {'wiek': [40,55], 'rok':[60,65]}
age = pd.DataFrame(data=data60_65)
print(nbrs.predict(age))


# Zadanie 3:

# In[18]:


wina = pd.read_csv("/Users/Komputer/Downloads/winequality-red.csv")


# In[19]:


wina.describe()


# In[ ]:





# Zadanie 4

# In[20]:


from sklearn.model_selection import train_test_split


# In[22]:


X = wina.drop(columns="jakosc")
y = wina.jakosc


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[24]:


nbrs = KNeighborsClassifier(n_neighbors=2, weights='uniform')
nbrs.fit(X_train,y_train)
print(f'dokladnosc zbioru testowego: {nbrs.score(X_train, y_train)}')
print(f'dokladnosc calego zbioru: {nbrs.score(X, y)}')


# In[25]:


nbrs = KNeighborsClassifier(n_neighbors=2, weights='distance')
nbrs.fit(X_train,y_train)
print(f'dokladnosc zbioru testowego: {nbrs.score(X_train, y_train)}')
print(f'dokladnosc calego zbioru: {nbrs.score(X, y)}')


# In[26]:


nbrs = KNeighborsClassifier(n_neighbors=4, weights='uniform')
nbrs.fit(X_train,y_train)
print(f'dokladnosc zbioru testowego: {nbrs.score(X_train, y_train)}')
print(f'dokladnosc calego zbioru: {nbrs.score(X, y)}')


# In[27]:


nbrs = KNeighborsClassifier(n_neighbors=4, weights='distance')
nbrs.fit(X_train,y_train)
print(f'dokladnosc zbioru testowego: {nbrs.score(X_train, y_train)}')
print(f'dokladnosc calego zbioru: {nbrs.score(X, y)}')


# In[28]:


nbrs = KNeighborsClassifier(n_neighbors=1, weights='distance')
nbrs.fit(X_train,y_train)
print(f'dokladnosc zbioru testowego: {nbrs.score(X_train, y_train)}')
print(f'dokladnosc calego zbioru: {nbrs.score(X, y)}')


# In[29]:


nbrs = KNeighborsClassifier(n_neighbors=1, weights='uniform')
nbrs.fit(X_train,y_train)
print(f'dokladnosc zbioru testowego: {nbrs.score(X_train, y_train)}')
print(f'dokladnosc calego zbioru: {nbrs.score(X, y)}')


# Zadanie 5 

# In[36]:


from sklearn.model_selection import ShuffleSplit


# In[37]:


def function(X, y, k, weights='uniform'):
    nbrs = KNeighborsClassifier(n_neighbors=k, weights=weights)
    nbrs.fit(X,y)
    print(nbrs.score(X,y))
    return nbrs


# In[38]:


ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)


# In[39]:


for_distance = {1:[], 2:[], 3:[], 4:[], 5:[]}
for_uniform = {1:[], 2:[], 3:[], 4:[], 5:[]}
cross_validation_index = 1
for train_index, test_index in ss.split(X):
    X_train = X[X.index.isin(train_index)]
    y_train = y[y.index.isin(train_index)]
    X_test = X[X.index.isin(test_index)]
    y_test = y[y.index.isin(test_index)]
    scores = []
    for i in range(1,6):
        nbrs = function(X_train,y_train,i,'distance')
        score = nbrs.score(X_test,y_test)
        for_distance[cross_validation_index] = scores
    for i in range(1,6):
        nbrs = function(X_train,y_train,i,'uniform')
        score = nbrs.score(X_test,y_test)
        scores.append(score)
        for_uniform[cross_validation_index] = scores
    cross_validation_index += 1


# In[40]:


for k,v in for_distance.items():
    average = 0.0
    for score in v:
        average += score
    average = average/ 5
    print(f'Srednia dokladnosc dla zbiorow testowych i dla n_neighbors={k} i weights=distance wynosi {average}')


# In[41]:


for k,v in for_uniform.items():
    average = 0
    for score in v:
        average += score
    average = average / 5
    print(f'Srednia dokladnosc dla zbiorow testowych i dla n_neighbors={k} i weights=uniform wynosi {average}')


# In[ ]:




