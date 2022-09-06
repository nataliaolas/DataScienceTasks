#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[3]:


zarobki = pd.read_csv("C:/Users/Komputer/Downloads/zarobki30tys.csv")


# In[4]:


print( zarobki.shape )


# First column - Earnings and the number of hours worked per week

# In[5]:


print( zarobki["praca_godz_tyg"].describe())


# In[6]:


print( zarobki["praca_godz_tyg"].mode())


# In[7]:


print( "### Liczba osób składająca się na całość danych" )
print( zarobki["waga_danych"].sum()/51 )


# In[8]:


print( "### Histogram samej liczby wierszy po atrybucie praca_godz_tyg" )
hist1 = zarobki["praca_godz_tyg"].value_counts().sort_index()
print( hist1 )
print()


# In[9]:


plt.bar(hist1.index, hist1, width=0.9, color="blue")
plt.xticks(np.arange(0, 100, step=10))
plt.ylabel("Liczba wierszy w danych", fontsize=12)
plt.xlabel("Liczba przepracowanych godzin w tyg", fontsize=12)
plt.show()


# In[10]:


print( "### Liczba osób w danej grupie lat edukacji z podziałem na zarobki" )
hist3 = zarobki.groupby(["zarobki","praca_godz_tyg"]).sum()
hist3 = hist3["waga_danych"]
print( round( (hist3/51) ) )
print()


# In[12]:


plt.bar( np.arange(1, hist3["<=50K"].index.size+1, step=1)-0.2 , hist3.loc[("<=50K")]/51000000, width=0.4, color="blue", label="<=50K")
plt.bar( np.arange(1, hist3[">50K"].index.size+1, step=1)+0.2 , hist3.loc[(">50K")]/51000000, width=0.4, color="red", label=">50K")
plt.legend()
plt.xticks(np.arange(0, 100, step=5))
plt.ylabel("Liczba osób w milionach", fontsize=12)
plt.xlabel("Liczba przepracowanycn godzin", fontsize=12)
plt.show()


# Second column- earnings and relations 

# In[51]:


print( zarobki["relacje"].describe())


# In[52]:


print( zarobki["relacje"].mode())


# In[53]:


print( zarobki["waga_danych"].sum()/51 )


# In[54]:


hist2 = zarobki["relacje"].value_counts().sort_index()
print( hist2 )


# In[55]:


plt.bar(hist2.index, hist2, width=0.9, color="blue")
labels = ['Husband', 'Not-in-family ', 'Other-relative ', 'Own-child ','Unmarried ','Wife' ]
plt.xticks(rotation =90)
plt.ylabel("Liczba wierszy w danych", fontsize=12)
plt.xlabel("Liczba ludzi w relacjach", fontsize=12)
plt.show()


# In[56]:


histrelacje = zarobki.groupby(["zarobki","relacje"]).sum()
histrelacje = histrelacje["waga_danych"]
print( round( (histrelacje/51) ) )
print()


# In[57]:


labels = ['Husband', 'Not-in-family ', 'Other-relative ', 'Own-child ','Unmarried ','Wife' ]
plt.bar( np.arange(1, histrelacje["<=50K"].index.size+1, step=1)-0.2 , histrelacje.loc[("<=50K")]/51000000, width=0.4, color="blue", label="<=50K")
plt.bar( np.arange(1, histrelacje[">50K"].index.size+1, step=1)+0.2 , histrelacje.loc[(">50K")]/51000000, width=0.4, color="red", label=">50K")
plt.legend()
plt.xticks(np.arange(1, histrelacje["<=50K"].index.size+1, step=1),labels,rotation =90)
plt.ylabel("Liczba osób w milionach", fontsize=12)
plt.xlabel("Liczba ludzi w danych relacjach", fontsize=12)
plt.show()


# Third column - earnings and gender

# In[5]:


print( zarobki["plec"].describe())


# In[6]:


print( zarobki["plec"].mode() )


# In[7]:


print( zarobki["waga_danych"].sum()/51 )


# In[8]:


hist_plec = zarobki["plec"].value_counts().sort_index()
print( hist_plec )


# In[9]:


plt.bar(hist_plec.index, hist_plec, width=0.9, color="blue")
labels = ['Female', 'Male' ]
plt.xticks(rotation =90)
plt.ylabel("Liczba wierszy w danych", fontsize=12)
plt.xlabel("Pleć", fontsize=12)
plt.show()


# In[10]:


histpl = zarobki.groupby(["zarobki","plec"]).sum()
histpl =  histpl["waga_danych"]
print( round( (histpl/51) ) )
print()


# In[58]:


labels = ['Female', 'Male' ]
plt.bar( np.arange(1, histpl["<=50K"].index.size+1, step=1)-0.2 , histpl.loc[("<=50K")]/51000000, width=0.4, color="blue", label="<=50K")
plt.bar( np.arange(1, histpl[">50K"].index.size+1, step=1)+0.2 , histpl.loc[(">50K")]/51000000, width=0.4, color="red", label=">50K")
plt.legend()
plt.xticks(np.arange(1, histpl["<=50K"].index.size+1, step=1),labels,rotation =90)
plt.ylabel("Liczba osób w milionach", fontsize=12)
plt.xlabel("Plec", fontsize=12)
plt.show()


# Fourth column -  earnings and employment

# In[19]:


print( zarobki["zatrudnienie"].describe())


# In[20]:


print( zarobki["zatrudnienie"].mode())


# In[21]:


print( zarobki["waga_danych"].sum()/51)


# In[22]:


hist_zatr = zarobki["zatrudnienie"].value_counts().sort_index()
print( hist_zatr )


# In[24]:


plt.bar(hist_zatr.index, hist_zatr, width=0.9, color="blue")
labels = ['Federal-gov','Local-gov','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay'  ]
plt.xticks(rotation =90)
plt.ylabel("Liczba wierszy w danych", fontsize=12)
plt.xlabel("Rodzaj zatrudnienia", fontsize=12)
plt.show()


# In[29]:


histzat = zarobki.groupby(["zarobki","zatrudnienie"]).sum()
histzat =  histzat["waga_danych"]
print( round( (histzat/51) ) )
print()


# In[32]:


labels = ['Federal-gov','Local-gov','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay'  ]
plt.bar( np.arange(1, histzat["<=50K"].index.size+1, step=1)-0.2 , histzat.loc[("<=50K")]/51000000, width=0.4, color="blue", label="<=50K")
plt.bar( np.arange(1, histzat[">50K"].index.size+1, step=1)+0.2 , histzat.loc[(">50K")]/51000000, width=0.4, color="red", label=">50K")
plt.legend()
plt.xticks(np.arange(1, histzat["<=50K"].index.size+1, step=1),labels,rotation =90)
plt.ylabel("Liczba osób w milionach", fontsize=12)
plt.xlabel("Rodzaj zatrudnienia", fontsize=12)
plt.show()


# Fifth column- employment and race

# In[33]:


print( zarobki["rasa"].describe())


# In[34]:


print( zarobki["rasa"].mode())


# In[35]:


print( zarobki["waga_danych"].sum()/51)


# In[36]:


hist_rasa = zarobki["rasa"].value_counts().sort_index()
print( hist_rasa )


# In[38]:


plt.bar(hist_rasa.index, hist_rasa, width=0.9, color="blue")
labels = ['Amer-Indian-Eskimo ','Asian-Pac-Islander','Black','Other','White ']
plt.xticks(rotation =90)
plt.ylabel("Liczba wierszy w danych", fontsize=12)
plt.xlabel("Rasa", fontsize=12)
plt.show()


# In[39]:


histrasa = zarobki.groupby(["zarobki","rasa"]).sum()
histrasa =  histrasa["waga_danych"]
print( round( (histrasa/51) ) )
print()


# In[40]:


labels = ['Amer-Indian-Eskimo ','Asian-Pac-Islander','Black','Other','White ']
plt.bar( np.arange(1, histrasa["<=50K"].index.size+1, step=1)-0.2 , histrasa.loc[("<=50K")]/51000000, width=0.4, color="blue", label="<=50K")
plt.bar( np.arange(1, histrasa[">50K"].index.size+1, step=1)+0.2 , histrasa.loc[(">50K")]/51000000, width=0.4, color="red", label=">50K")
plt.legend()
plt.xticks(np.arange(1, histrasa["<=50K"].index.size+1, step=1),labels,rotation =90)
plt.ylabel("Liczba osób w milionach", fontsize=12)
plt.xlabel("Rodzaj rasy", fontsize=12)
plt.show()


# In[ ]:




