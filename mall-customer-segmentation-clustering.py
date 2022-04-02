#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Mall Customer Segmentation - Clustering Method</h1>

# **Importing Modules**

# In[ ]:

import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# **Importing Data**

# In[2]:


st = pd.read_csv('../Mall_Customers.csv')
st.shape


# **How our data looks...**

# In[3]:


st.head()


# In[4]:


st.columns


# **Lets aggregate values based on Gender**

# In[5]:


st.groupby('Genre')['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# *Average Spending Score of Females is more than Males.*

# **Let's convert Genre column to numerical**

# In[6]:


st.Genre = st.Genre.map({'Female':1,'Male':2})


# ### Building Clustering model

# In[7]:


from mpl_toolkits.mplot3d import Axes3D


# In[8]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = st['Age']
y = st['Annual Income (k$)']
z = st['Spending Score (1-100)']

ax.set_xlabel("Happiness")
ax.set_ylabel("Economy")
ax.set_zlabel("Health")

ax.scatter(x, y, z)

plt.show()


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


X = st.drop(['CustomerID','Genre'],axis=1)


# In[11]:


X.sample()


# In[12]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
center_1 = kmeans.cluster_centers_
print(center_1)


# **Let's Visualize the Two Clusters**

# In[13]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = st['Age']
y = st['Annual Income (k$)']
z = st['Spending Score (1-100)']

ax.set_xlabel("Happiness")
ax.set_ylabel("Economy")
ax.set_zlabel("Health")

ax.scatter(x, y, z, c=kmeans.labels_)
ax.scatter(center_1[:,0],center_1[:,1],center_1[:,2], color = 'Red')

plt.show()


# **We will make 10 Clusters Now**

# In[14]:


kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
center_2 = kmeans.cluster_centers_
print(center_2)


# In[15]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = st['Age']
y = st['Annual Income (k$)']
z = st['Spending Score (1-100)']

ax.set_xlabel("Happiness")
ax.set_ylabel("Economy")
ax.set_zlabel("Health")

ax.scatter(x, y, z, c=kmeans.labels_)
ax.scatter(center_2[:,0],center_2[:,1],center_2[:,2], color = 'Red')

plt.show()


# **Let's Plot the Elbow curve to find out the ideal Number of Clusters**

# In[16]:


from sklearn.cluster import KMeans
l=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X)
    l.append(kmeans.inertia_)
plt.plot(range(1,11),l)
plt.title('Elbow Curve')
plt.show() 


# *Six Clusters We can have as ideal Clusters*

# In[17]:


kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
center_3 = kmeans.cluster_centers_
print(center_3)


# In[18]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = st['Age']
y = st['Annual Income (k$)']
z = st['Spending Score (1-100)']

ax.set_xlabel("Happiness")
ax.set_ylabel("Economy")
ax.set_zlabel("Health")

ax.scatter(x, y, z, c=kmeans.labels_)
ax.scatter(center_3[:,0],center_3[:,1],center_3[:,2], color = 'Red')

plt.show()


# In[19]:


kmeans.labels_


# In[20]:


st['Customer Segment'] = kmeans.labels_


# In[21]:


st.sample(10)


# # Thank You 
# 
# #### Please Mention your Feedback
