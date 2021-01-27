#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix


# In[2]:


data_raw = pd.read_csv('C:\\Users\\CSE_JUST\\Desktop\exported_data.csv',nrows=300)
data_f = pd.DataFrame(data_raw)
data_f


# In[14]:


pd.set_option('display.max_colwidth', 190)
data_f['ingredients']


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf =TfidfVectorizer(max_df=1.0,analyzer = 'word',ngram_range = (1,5),min_df = 0,stop_words = 'english')


# In[11]:


user_pref = input("")
data_f['Restrict_food_ingds'] = user_pref
data_f['cleaned'] = data_f['cleaned'].str.replace('\d+', '')
d_table = pd.DataFrame(data_f)
d_table


# In[16]:


d_table["Restrict_food_ingds"]


# In[6]:


fd_vector = tf.fit_transform(data_f['cleaned'])
print(fd_vector.shape)
print(tf.vocabulary_)


# In[ ]:





# In[7]:


vector = tf.get_feature_names()
vector


# In[8]:


#print(type(fd_vector))
print(fd_vector.todense())


# In[9]:


dense = fd_vector.todense()
denselist = dense.tolist()
df = pd.DataFrame(
    denselist,columns=tf.get_feature_names())


# In[ ]:




