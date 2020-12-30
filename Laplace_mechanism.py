#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import torch
torch.uint8


# In[2]:


def create_neighbors(db, index_to_remove):
    return torch.cat((db[0:index_to_remove], db[index_to_remove+1:]));


# In[3]:


#bins=np.arange(min(data), max(data) + 1, 1)
#print(ipums['Income'].count())
#plt.hist(ipums['Income'], bins)

ipums = pd.read_csv("ipums_h2.csv")


IncomeH200 = []
IncomeH400 = []
IncomeH600 = []
IncomeH800 = []
IncomeH1000 = []


IncomeH = ipums['Income']
plt.hist(IncomeH, bins = [0,200,400,600,800,1000]) 
for i in range(len(IncomeH)):
    if (IncomeH[i] <= 20):
        IncomeH200.append(IncomeH[i])
    elif (IncomeH[i] <= 40):
        IncomeH400.append(IncomeH[i])
    elif (IncomeH[i] <= 60):
        IncomeH600.append(IncomeH[i])
    elif (IncomeH[i] <= 80):
        IncomeH800.append(IncomeH[i])
    else:
        IncomeH1000.append(IncomeH[i])
plt.title("Original Income Histogram")
plt.ylabel('Frequency')
plt.xlabel('Income')

plt.show()

AgeH20 = []
AgeH40 = []
AgeH60 = []
AgeH80 = []
AgeH100 = []

AgeH = ipums['Age']
plt.hist(AgeH, bins = [0,20,40,60,80,100]) 
for i in range(len(AgeH)):
    if (AgeH[i] <= 20):
        AgeH20.append(AgeH[i])
    elif (AgeH[i] <= 40):
        AgeH40.append(AgeH[i])
    elif (AgeH[i] <= 60):
        AgeH60.append(AgeH[i])
    elif (AgeH[i] <= 80):
        AgeH80.append(AgeH[i])
    else:
        AgeH100.append(AgeH[i])
plt.title("Original Age Histogram")
plt.ylabel('Frequency')
plt.xlabel('Age')
plt.show()


GenderHH2 = []
GenderHH1 = []
GenderH = ipums['Gender']
plt.hist(GenderH, bins = [1,2,3]) 
for i in range(len(GenderH)):
    if (GenderH[i] > 1):
        GenderHH2.append(GenderH[i])
    else:
        GenderHH1.append(GenderH[i])
plt.title("Original Gender Histogram")
plt.ylabel('Frequency')
plt.xlabel('Gender')
plt.show()


# In[4]:


IncomeH2 = torch.FloatTensor(IncomeH)
AgeH2 = torch.FloatTensor(AgeH)
GenderH2 = torch.FloatTensor(GenderH)


# In[5]:


def get_neighboring_databases(db):
    neighboring_databases = list()
    for i in range(len(db)):
        ndb = create_neighbors(db,i)
        neighboring_databases.append(ndb)
    return neighboring_databases  


# In[6]:


IncomeH3 = get_neighboring_databases(IncomeH2)
IncomeH3


# In[7]:


AgeH3 = get_neighboring_databases(AgeH2)
AgeH3


# In[8]:


GenderH3 = get_neighboring_databases(GenderH2)
GenderH3


# In[9]:


epsilon =0.1


# In[10]:


def count_query(db):
    return db.numel()


# In[11]:


def laplace_mechanism(db,query, sensitivity):
    beta = sensitivity /epsilon
    noise = torch.tensor(np.random.laplace(0,beta,1))
    return query(db) + noise


# In[12]:


global_sensitivity = 0
qdb = count_query(AgeH2)
for ndb in AgeH3:
    q_ndb = count_query(ndb)
    print(q_ndb , qdb)
    q_distance = abs(q_ndb - qdb)
    print(q_distance)
    if(q_distance > global_sensitivity):
        global_sensitivity = q_distance


# In[13]:


global_sensitivity


# In[14]:


lapI = laplace_mechanism(IncomeH2, count_query, global_sensitivity)
lapI


# In[15]:


lapA = laplace_mechanism(AgeH2, count_query, global_sensitivity)
lapA


# In[16]:


lapG = laplace_mechanism(GenderH2, count_query, global_sensitivity)
lapG


# In[17]:


while lapG > 10:
    lapG = lapG % 10
lapG2 = int(lapG)
lapG3 = lapG-lapG2
lapG3


# In[18]:


IncomeD = []
for el in IncomeH2:
    el = IncomeH2 + lapI
    el.tolist()
    el = el - 20000
    print(el)
    IncomeD.append(el)


# In[19]:


AgeD = []
for el in AgeH2:
    el = AgeH2 + lapA
    el.tolist()
    el = el - 20000
    print(el)
    AgeD.append(el)


# In[20]:


GenderD = []
for el in GenderH2:
    el = GenderH2 + lapG3
    el.tolist()
    #el = el - 20000
    print(el)
    GenderD.append(el)


# In[21]:


IncomeD200 = []
IncomeD400 = []
IncomeD600 = []
IncomeD800 = []
IncomeD1000 = []

plt.hist(IncomeD[0], bins = [0,200,400,600,800,1000])

for i in range(len(IncomeD[0])):
    if (IncomeD[0][i] <= 200):
        IncomeD200.append(IncomeD[0][i])
    elif (IncomeD[0][i] <= 400):
        IncomeD400.append(IncomeD[0][i])
    elif (IncomeD[0][i] <= 600):
        IncomeD600.append(IncomeD[0][i])
    elif (IncomeD[0][i] <= 800):
        IncomeD800.append(IncomeD[0][i])
    else:
        IncomeD1000.append(IncomeD[0][i])
plt.title("differintal Income Histogram")
plt.ylabel('Frequency')
plt.xlabel('Income')
plt.show()


# In[22]:


AgeD20 = []
AgeD40 = []
AgeD60 = []
AgeD80 = []
AgeD100 = []
plt.hist(AgeD[0], bins = [0,20,40,60,80,100])
for i in range(len(AgeD[0])):
    if (AgeD[0][i] <= 20):
        AgeD20.append(AgeD[0][i])
    elif (AgeD[0][i] <= 40):
        AgeD40.append(AgeD[0][i])
    elif (AgeD[0][i] <= 60):
        AgeD60.append(AgeD[0][i])
    elif (AgeD[0][i] <= 80):
        AgeD80.append(AgeD[0][i])
    else:
        AgeD100.append(AgeD[0][i])
plt.title("differintal Age Histogram")
plt.ylabel('Frequency')
plt.xlabel('Age')
plt.show()


# In[23]:


GenderD1 = []
GenderD2 = []
plt.hist(GenderD[0], bins = [1,2,3]) 

for i in range(len(GenderD[0])):
    if (GenderD[0][i] > 1):
        GenderD2.append(GenderD[0][i])
    else:
        GenderD1.append(GenderD[0][i])
plt.title("differintal Gender Histogram")
plt.ylabel('Frequency')
plt.xlabel('Gender')
plt.show()


# # Task2

# In[24]:


#Mse for Gender
difference_array1 = np.subtract(len(GenderHH1), len(GenderD1))
difference_array2 = np.subtract(len(GenderHH2), len(GenderD2))
squared_array1 = np.square(difference_array1)
squared_array2 = np.square(difference_array2)
nsum = squared_array1 + squared_array2
mse = nsum / 3

print(mse)


# In[25]:


#Mse for Age
difference_array1 = np.subtract(len(AgeH20), len(AgeD20))
difference_array2 = np.subtract(len(AgeH40), len(AgeD40))
difference_array3 = np.subtract(len(AgeH60), len(AgeD60))
difference_array4 = np.subtract(len(AgeH80), len(AgeD80))
difference_array5 = np.subtract(len(AgeH100), len(AgeD100))
squared_array1 = np.square(difference_array1)
squared_array2 = np.square(difference_array2)
squared_array3 = np.square(difference_array3)
squared_array4 = np.square(difference_array4)
squared_array5 = np.square(difference_array5)
nsum = squared_array1 + squared_array2 + squared_array3 + squared_array4 + squared_array5
mse = nsum / 6

print(mse)


# In[26]:


#Mse for Income
difference_array1 = np.subtract(len(IncomeH200), len(IncomeD200))
difference_array2 = np.subtract(len(IncomeH400), len(IncomeD400))
difference_array3 = np.subtract(len(IncomeH600), len(IncomeD600))
difference_array4 = np.subtract(len(IncomeH800), len(IncomeD800))
difference_array5 = np.subtract(len(IncomeH1000), len(IncomeD1000))
squared_array1 = np.square(difference_array1)
squared_array2 = np.square(difference_array2)
squared_array3 = np.square(difference_array3)
squared_array4 = np.square(difference_array4)
squared_array5 = np.square(difference_array5)
nsum = squared_array1 + squared_array2 + squared_array3 + squared_array4 + squared_array5
mse = nsum / 6

print(mse)

