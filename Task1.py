#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd


# Ex 1

# In[95]:


zarobki = pd.read_csv("/Users/Komputer/Downloads/zarobki200.csv")


# In[45]:


zarobki.describe()


# In[46]:


zarobki.wiek.mean()


# In[47]:


zarobki.praca_godz_tyg.max()


# In[48]:


zarobki.praca_godz_tyg.min()


# In[49]:


zarobki.loc[0:11,["rasa","plec","kraj"]]


# In[50]:


zarobki.to_html("/Users/Komputer/Downloads/zarobki200.html")


# In[51]:


lista = zarobki.loc[80:100,"zatrudnienie":"zawod"]


# In[52]:


print(lista)


# Ex 2:

# In[53]:


mexicos = zarobki[zarobki["kraj"]=="Mexico"]


# In[54]:


mexicos


# In[55]:


len(mexicos)


# In[56]:


zarobki = zarobki.replace('Female','K')


# In[57]:


zarobki = zarobki.replace("Male","M")


# In[58]:


zarobki = zarobki.rename(columns={"plec":"K/M"})


# In[59]:


zarobki.tail(20)


# In[60]:


zarobki_wieksze_od_50K = zarobki[zarobki["zarobki"]==">50K"]


# In[92]:


zarobki_wieksze_od_50K.to_json("/Users/Komputer/Downloads/zarobki200.json",orient="index") 


# In[62]:


zarobki.groupby(by=["zawod"]).size()


# Ex 3:

# In[63]:


rows = zarobki.sample(n=15)


# In[64]:


rows


# In[65]:


zarobki_zadanie = zarobki.drop(columns=['zatrudnienie','stan_cywilny','zawod','relacje','rasa','K/M','kraj'])


# In[66]:


zarobki_zadanie


# In[67]:


zarobki_zadanie[zarobki_zadanie['zarobki']=='>50K'].query('wiek < 35')


# In[68]:


zarobki


# Ex 4:

# In[69]:


zarobki["zatrudnienie_zawod"] = zarobki["zatrudnienie"] + ":"+ zarobki["zarobki"]


# In[70]:


zarobki


# In[71]:


zarobki = zarobki.drop(columns=['zatrudnienie','zawod']) 


# In[82]:


zarobki['przepracowane_godz'] = (zarobki['wiek'] - (7 - zarobki['edukacja_lata']))*52*zarobki['praca_godz_tyg']


# In[88]:


zarobki.sort_values(by=['przepracowane_godz'], ascending =False).to_csv('./przepracowane_godz.csv', index=False)


# Ex 5:

# In[84]:


import numpy as np
zarobki_wiek_bezpustych = zarobki['wiek'].replace(np.nan,0)
zarobki['wiek'] = zarobki['wiek'].replace(np.nan,0)


# In[85]:


zarobki_wiek_bezpustych = zarobki_wiek_bezpustych[zarobki_wiek_bezpustych != 0]
zarobki['wiek'] = zarobki['wiek'].replace(0,round(zarobki_wiek_bezpustych.mean(),0))


# In[86]:


zarobki_praca_godz_tyg_bez_pustych = zarobki['praca_godz_tyg'].replace(np.nan,0)
zarobki_praca_godz_tyg_bez_pustych = zarobki_praca_godz_tyg_bez_pustych[zarobki_praca_godz_tyg_bez_pustych != 0]
zarobki['praca_godz_tyg'] =  zarobki['praca_godz_tyg'].replace(np.nan,0)


min_praca_godz_tyg = int(zarobki_praca_godz_tyg_bez_pustych.min())
max_praca_godz_tyg = int(zarobki_praca_godz_tyg_bez_pustych.max())

randomowa_praca_godz_choice=np.random.choice(range(min_praca_godz_tyg,max_praca_godz_tyg),1, replace=False)
randomowa_praca_godz_sample = zarobki[zarobki['praca_godz_tyg'] != 0].sample(1)
print(int(randomowa_praca_godz_choice))
print(int(randomowa_praca_godz_sample['praca_godz_tyg']))

zarobki['praca_godz_tyg'] = zarobki['praca_godz_tyg'].replace(0,int(randomowa_praca_godz_choice))


# In[87]:


zarobki['edukacja_lata'] = zarobki['edukacja_lata'].replace(np.nan,zarobki['edukacja_lata'].mode()[0])


# In[90]:


zarobki

