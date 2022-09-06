#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[3]:


irys = pd.read_csv("/Users/Komputer/Downloads/iris.csv", names=["SL","SW","PL","PW","IrCl"])


# Exercise 1:

# In[4]:


X = irys[["SL","SW","PL","PW"]]
Y = irys["IrCl"]


# In[5]:


plt.scatter(X["SL"][0:50] ,X["SW"][0:50], color="blue")
plt.scatter(X["SL"][50:100],X["SW"][50:100],color="orange")
plt.scatter(X["SL"][100:150],X["SW"][100:150], color="green")


# Exercise 2:

# In[6]:


plt.scatter(X["SL"][0:50] ,X["SW"][0:50], color="blue", label="Setosa")
plt.scatter(X["SL"][50:100],X["SW"][50:100],color="orange", label="Versicolor")
plt.scatter(X["SL"][100:150],X["SW"][100:150], color="green", label="Virginica")
plt.ylabel("Sepal width")
plt.xlabel("Sepal length")
plt.suptitle("Zależność SL:SW",fontweight="bold")
plt.legend()


# Exercise 3:

# Sepal length - Sepal width

# In[7]:


plt.scatter(X["SL"][0:50] ,X["SW"][0:50], color="red", label="Setosa")
plt.scatter(X["SL"][50:100],X["SW"][50:100],color="green", label="Versicolor")
plt.scatter(X["SL"][100:150],X["SW"][100:150], color="blue", label="Virginica")
plt.ylabel("Sepal width")
plt.xlabel("Sepal length")
plt.legend()


# Sepal length - Petal length

# In[8]:


plt.scatter(X["SL"][0:50] ,X["PL"][0:50], color="red", label="Setosa")
plt.scatter(X["SL"][50:100],X["PL"][50:100],color="green", label="Versicolor")
plt.scatter(X["SL"][100:150],X["PL"][100:150], color="blue", label="Virginica")
plt.ylabel("Petal length")
plt.xlabel("Sepal length")
plt.legend()


# Sepal length - Petal width

# In[9]:


plt.scatter(X["SL"][0:50] ,X["PW"][0:50], color="red", label="Setosa")
plt.scatter(X["SL"][50:100],X["PW"][50:100],color="green", label="Versicolor")
plt.scatter(X["SL"][100:150],X["PW"][100:150], color="blue", label="Virginica")
plt.ylabel("Petal width")
plt.xlabel("Sepal length")
plt.legend()


# Sepal width - Petal length

# In[10]:


plt.scatter(X["SW"][0:50] ,X["PL"][0:50], color="red", label="Setosa")
plt.scatter(X["SW"][50:100],X["PL"][50:100],color="green", label="Versicolor")
plt.scatter(X["SW"][100:150],X["PL"][100:150], color="blue", label="Virginica")
plt.ylabel("Petal length")
plt.xlabel("Sepal width")
plt.legend()


# Sepal width - Petal width

# In[11]:


plt.scatter(X["SW"][0:50] ,X["PW"][0:50], color="red", label="Setosa")
plt.scatter(X["SW"][50:100],X["PW"][50:100],color="green", label="Versicolor")
plt.scatter(X["SW"][100:150],X["PW"][100:150], color="blue", label="Virginica")
plt.ylabel("Petal width")
plt.xlabel("Sepal width")
plt.legend()


# Petal length - Petal width

# In[12]:


plt.scatter(X["PL"][0:50] ,X["PW"][0:50], color="red", label="Setosa")
plt.scatter(X["PL"][50:100],X["PW"][50:100],color="green", label="Versicolor")
plt.scatter(X["PL"][100:150],X["PW"][100:150], color="blue", label="Virginica")
plt.ylabel("Petal width")
plt.xlabel("Petal legnth")
plt.legend()


# Exercise 4:

# In[13]:


def wykres_zaleznosc(x,y, x_name, y_name):
    plt.scatter(irys[irys['IrCl'] == 'Iris-setosa'][x], irys[irys['IrCl'] == 'Iris-setosa'][y],label="Setosa")
    plt.scatter(irys[irys['IrCl'] == 'Iris-virginica'][x], irys[irys['IrCl'] == 'Iris-virginica'][y], label="Virginica")
    plt.scatter(irys[irys['IrCl'] == 'Iris-versicolor'][x], irys[irys['IrCl'] == 'Iris-versicolor'][y], label="Versicolor")
    plt.legend() 
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    zaleznosc = "Zaleznosc " + x + ":" + y
    plt.title(zaleznosc)


# In[14]:


plt.subplot(3,2,1)
wykres_zaleznosc('SL','SW','Sepal length', 'Sepal width')
plt.subplot(3,2,2)
wykres_zaleznosc('SL','PL','Sepal length', 'Petal length')
plt.subplot(3,2,3)
wykres_zaleznosc('SL','PW','Sepal length', 'Petal width')
plt.subplot(3,2,4)
wykres_zaleznosc('SW','PL','Sepal width', 'Petal length')
plt.subplot(3,2,5)
wykres_zaleznosc('SW','PW','Sepal width', 'Petal width')
plt.subplot(3,2,6)
wykres_zaleznosc('PL','PW','Petal length', 'Petal width')

plt.show()


# Exercise 5:

# In[19]:


irys.describe()


# In[16]:


irys.groupby(by='IrCl').describe()


# In[17]:


ySL = np.array([5.01, 5.94, 6.59])
ySW = np.array([3.42, 2.77, 2.97])
yPL = np.array([1.46, 4.26, 5.55])
yPW = np.array([0.24, 1.33, 2.03])
index = ['Setosa', 'Verginica', 'Versi-Color']

df = pd.DataFrame({'Sepal Length' : ySL, 'Sepal Width' : ySW, 'Petal Length' : yPL, 'Petal Width' : yPW}, index=index)


# In[18]:


df.plot.bar(rot=0)


# In[ ]:





# In[ ]:




