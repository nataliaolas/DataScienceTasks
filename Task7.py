#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pydotplus


# In[2]:


pip install graphviz


# In[ ]:


conda install python-graphviz


# In[1]:


import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.tree import export_text
import numpy as np
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# Ex 1:

# In[ ]:


dane = pandas.read_csv("/Users/Komputer/Downloads/zoo.csv",names=["nazwa","włosy","pierze","jajka","mleko","latajacy","wodny","drapieznik","uzebiony",
"szkielet","oddycha","jadowity","pletwy","nogi","ogon","udomowiony","rozmiar_kota","typ"])


# In[ ]:


features=["włosy","pierze","jajka","mleko","latajacy","wodny","drapieznik","uzebiony","szkielet","oddycha","jadowity","pletwy","nogi","ogon","udomowiony","rozmiar_kota"]


# In[ ]:


X = dane[features]dane[features]
y = dane["typ"]


# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


dtree = dtree.fit(X,y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')


# In[ ]:


img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


r = export_text(dtree,feature_names=features)
print(r)


# In[ ]:


zaskroniec = ([0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0])
zaskroniec = np.array([0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0])
zaskroniec = np.reshape(zaskroniec,(1,-1))
print(dtree.predict(zaskroniec))


# Ex 2:

# In[ ]:


dtree_gini = DecisionTreeClassifier(criterion="gini",random_state=None)


# In[ ]:


dtree_gini = dtree_gini.fit(X,y)


# In[ ]:


data = tree.export_graphviz(dtree_gini, out_file=None, feature_names=features,filled=True)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('gini.png')


# In[ ]:


img=pltimg.imread('gini.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


r = export_text(dtree_gini,feature_names=features)
print(r)


# In[ ]:


dtree_entropy = DecisionTreeClassifier(criterion="entropy",random_state=None)


# In[ ]:


dtree_entropy = dtree_entropy.fit(X,y)


# In[ ]:


data = tree.export_graphviz(dtree_entropy, out_file=None, feature_names=features,filled=True)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('entropy.png')


# In[ ]:


img=pltimg.imread('entropy.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


r = export_text(dtree_entropy,feature_names=features)
print(r)


# In[ ]:


dzdzownica = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
dzdzownica = np.reshape(dzdzownica,(1,-1))
print(dtree.predict(dzdzownica))


# In[ ]:


kangur = np.array([1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0])
kangur = np.reshape(kangur,(1,-1))
print(dtree.predict(kangur))


# In[ ]:


motyl = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
motyl = np.reshape(motyl,(1,-1))
print(dtree.predict(motyl))


# Ex 3:

# In[2]:


wina = pandas.read_csv("/Users/Komputer/Downloads/winequality-white.csv")


# In[3]:


wina.describe()


# Ex 4:

# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


features=["stała kwasowość","kwasowość lotna","kwas cytrynowy","cukier resztkowy","chlorki","wolny dwutlenek siarki","całkowity dwutlenek siarki","gęstość","pH","siarczany","alcohol"]


# In[6]:


X = wina[features]
y = wina["jakosc"]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[8]:


dtree_gini = DecisionTreeClassifier(criterion="gini",random_state=None)


# In[9]:


dtree_gini = dtree_gini.fit(X_train,y_train)


# In[10]:


print(dtree_gini.score(X_test,y_test))


# In[11]:


data = tree.export_graphviz(dtree_gini, out_file=None, feature_names=features,filled=True)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('wina_gini.png')


# In[12]:


img=pltimg.imread('wina_gini.png')
imgplot = plt.imshow(img)
plt.show()


# In[13]:


r = export_text(dtree_gini,feature_names=features)
print(r)


# In[24]:


dtree_gini.get_n_leaves()


# In[14]:


dtree_entropy = DecisionTreeClassifier(criterion="entropy",random_state=None)


# In[17]:


dtree_entropy = dtree_entropy.fit(X_train,y_train)


# In[18]:


print(dtree_entropy.score(X_test,y_test))


# In[19]:


data = tree.export_graphviz(dtree_entropy, out_file=None, feature_names=features,filled=True)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('wina_entropy.png')


# In[20]:


img=pltimg.imread('wina_entropy.png')
imgplot = plt.imshow(img)
plt.show()


# In[21]:


r = export_text(dtree_gini,feature_names=features)
print(r)


# In[ ]:




