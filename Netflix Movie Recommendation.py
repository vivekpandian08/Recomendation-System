#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('C:/Users/vivek/OneDrive/Desktop/netflix')


# In[2]:


import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import *
sns.set_style("darkgrid")


# In[3]:


np.random.seed(2020)


# Loading the file

# In[7]:


if not os.path.isfile('data1.csv'):
    # Create a file 'data.csv' before reading it
    # Read all the files in netflix and store them in one big file('data.csv')
    data = open('data1.csv', mode='w')
    
    row = list()
    files=['combined_data_1.txt']
    for file in files:
        print("Reading ratings from {}...".format(file))
        with open(file) as f:
            for line in f: 
                del row[:] # you don't have to do this.
                line = line.strip()
                if line.endswith(':'):
                    # All below are ratings for this movie, until another movie appears.
                    movie_id = line.replace(':', '')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(0, movie_id)
                    data.write(','.join(row))
                    data.write('\n')
        print("Done.\n")


# In[8]:


df = pd.read_csv('data1.csv', sep=',', names=['movie_id', 'customer_id','rating','date'])
df.date = pd.to_datetime(df.date)


# In[9]:


print ("The data file has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " rows")


# In[10]:


#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(15)


# In[11]:


df.info()


# In[12]:


for column in df.columns:
    print(f"{column}: Number of unique values {df[column].nunique()}")
    print(f"Number of Null values: {df[column].isnull().sum()}")
    print("----------------------------------------------")


# In[10]:


class_count1 = df['rating'].value_counts()
sns.set(style="darkgrid")
sns.barplot(class_count1.index, class_count1.values, alpha=0.9)
plt.title('Frequency Distribution of Rating')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Rating', fontsize=12)
plt.show()
print(df['rating'].value_counts())
(df.rating.value_counts(normalize=True))*100


# In[11]:


df.head()


# In[12]:


print("Total data ")
print("-"*50)
print("\nTotal Number of ratings :",df.shape[0])
print("Total Number of Users   :", len(np.unique(df.customer_id)))
print("Total Number of movies  :", len(np.unique(df.movie_id)))


# In[13]:


import random
random.seed(1) 
df3=df.sample(100000,random_state=99)


# In[14]:


df2= pd.read_csv('movie_titles.csv', 
                           encoding = 'ISO-8859-1', 
                           header = None, 
                           names = ['movie_id', 'year', 'movie_name'])


# In[15]:


df1=pd.merge(df3,df2,on="movie_id")


# In[16]:


df1.head()


# In[17]:


#Creating Pivot Table:
df_p = pd.pivot_table(df1,values='rating',index='customer_id',columns='movie_name')


# In[18]:


df_p.head()


# In[ ]:


corrMatrix = df_p.corr(method='pearson', min_periods=100)


# In[ ]:


userId = 2
print("Movies and their ratings for user ", userId)


# In[ ]:


myRatings = df_p.loc[customer_id].dropna()    
#loc = indexed location. dropna() = drop all NaN values. Filters out all movies for which user with userId didn't give any rating


# In[ ]:


sim= pd.Series()


# In[ ]:


for i in range(0, len(myRatings.index)):
    sims = corrMatrix[myRatings.index[i]].dropna()
    sims = sims.map(lambda x: x * myRatings[i])
    sim = sim.append(sims)


# In[ ]:


sim.sort_values(inplace = True, ascending = False)

sim = simCandidates.groupby(simCandidates.index).sum()

sim.sort_values(inplace = True, ascending = False)
filteredSims = sim.drop(myRatings.index, errors='ignore')
print(filteredSims.head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




