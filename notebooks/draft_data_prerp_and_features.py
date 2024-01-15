#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the neccesary libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif


# In[3]:


# Installing the data path for our input and output data


# In[4]:


raw_input_data_path = "/Users/ilyashnip/Documents/GitHub/ML_project_in_Finance_I_Signing_of_deposit_agreement/ML_project_in_Finance_I_Signing_of_deposit_agreement/Data/Input"
output_data_path = "/Users/ilyashnip/Documents/GitHub/ML_project_in_Finance_I_Signing_of_deposit_agreement/ML_project_in_Finance_I_Signing_of_deposit_agreement/Data/Output"


# In[5]:


# Data extraction


# In[6]:


file_path = f"{raw_input_data_path}/bank-additional-full.csv"


# In[7]:


df = pd.read_csv(file_path , sep = ';')


# In[8]:


df.head(15)


# In[9]:


# Data has been extracted. Now checking the names of colums.


# In[10]:


print(df.columns)


# In[11]:


# Deciding to delete some columns, such as 'contact', 'month', 'day_of_week'
# The column default will be deleted as well, because there is only 3 yes, all the others is now


# In[12]:


df.drop(columns=['contact', 'month', 'day_of_week', 'default'],  inplace=True,)
df.tail(15)


# In[13]:


df.shape


# In[14]:


# Checking the results, and seeing that amount of columns reduced from 21 to 17. 
# No more such columns as 'contact', 'month', 'day_of_week'


# In[15]:


#Now, lets take a look on what types of variables do we have in our data


# In[16]:


df.info()


# In[17]:


# Let's check the amout of data which are missing.


# In[18]:


missing_values = df.isna().sum()
print (missing_values)


# In[19]:


# No missing data in our data set.So far so good. 
# But data type of some of our columns is object. So we need to check them. 
# The process of checking the data will be the same for all of them:
# 1) Check what unique values. If something unpleasent will be found, than:
# 2) Data Cleaning


# In[20]:


#Starting from the column 'job'. Let's check:


# In[21]:


unique_values_job = df['job'].unique()
print(unique_values_job)


# In[22]:


# As a reluts, we can see that some of our rows include job:unknown. 
# In other words we have found that some data are missing.
# Let's clean it!


# In[23]:


df['job'] = df['job'].replace({'unknown': np.nan})
df.dropna(axis=0, inplace=True)
df.shape


# In[24]:


# After the cleaning we have deleted 330 rows, less than 1%.


# In[25]:


# Doing the same for marital.


# In[26]:


unique_values_marital = df['marital'].unique()
print(unique_values_marital)


# In[27]:


# And again, we can see that some data missing ('unknown').


# In[28]:


df['marital'] = df['marital'].replace({'unknown': np.nan})
df.dropna(axis=0, inplace=True)
df.shape


# In[29]:


# After the cleaning we have deleted 71 rows, less than 1%.


# In[30]:


unique_values_ed = df['education'].unique()
print(unique_values_ed)


# In[31]:


# And again, we can see that some data missing ('unknown').


# In[32]:


df['education'] = df['education'].replace({'unknown': np.nan})
df.dropna(axis=0, inplace=True)
df.shape


# In[33]:


# Once an accident, twice a coincidence, three times is a pattern.
# From now we can assume, that other columns will also have some missing varibales under the name "unknown"
# So, we will speed up our job a littel bit. By finnding and deleting all such data in all remain columns.


# In[34]:


df['housing'] = df['housing'].replace({'unknown': np.nan})
df['loan'] = df['loan'].replace({'unknown': np.nan})

df.dropna(axis=0, inplace=True)
df.shape


# In[35]:


# After the cleaning we have left with 38245 rows. 
# In totat, we have deleted 10700 rows of data or 26% of our original data set. 


# In[36]:


# In previous steps of data preperation, we haven't check the target variable "y"
# So let's check how many unique values there.


# In[37]:


unique_values_y = df['y'].unique()
print(unique_values_y)


# In[38]:


# On this stage, we know that our target variable 'y' is binary. Only yes=1 or no=0. No other options.
# Great, no need to delete some more data.

# Let's check how many of yes or now in the column.


# In[39]:


count_yes = df['y'].eq('yes').sum()
count_no = df['y'].eq('no').sum()
print(f'Number of "yes": {count_yes}')
print(f'Number of "no": {count_no}')


# In[40]:


# Vizualization of the results.


# In[41]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x = 'y', data = df)


# In[42]:


# We have found a problem. Our data set is unbalanced. 
# The number of "yes" answers is small. Almoust 7 times smaller than number of answers "no".
# Such great difference would cause a problems. 
# We can solve this problem by using weigths during the process of modeling.
# For a now, lets keep it, as it is.


# In[43]:


#Let's check the other categorical columns, maybe we will find the same problem as well.
#Starting from the 'job'


# In[44]:


sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x = 'job', data = df)


# In[45]:


# A lot of admin,technician and blue color jobs.


# In[46]:


sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x = 'marital', data = df)


# In[47]:


sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x = 'education', data = df)


# In[48]:


# Looks like is a very few of illiterate clients. Maybe will be better to eliminate them all?


# In[49]:


count_illiterate = df['y'].eq('illiterate').sum()
print(f'Number of "illiterate clients": {count_illiterate}')


# In[50]:


# That's it, after the previous manipulations all client's are have some education level. Delete the illiterate.


# In[51]:


df = df[df['education'] != 'illiterate']


# In[52]:


sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x = 'education', data = df)


# In[53]:


# Now the graph looks better.


# In[54]:


sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x = 'housing', data = df)


# In[55]:


sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x = 'loan', data = df)


# In[56]:


df.describe()


# In[57]:


#The graph below shows that all the features do not have a linear relationship. 
#Thanks to the “duration” attribute, you can notice that there is a fuzzy but boundary separating the target attribute when people answered “yes” or “no”. 
#This is logical, because if a person listens to a deposit offer for a long time, then he is most likely interested in the offer. 
#In other signs the connection is poorly visible. This graph also shows that all distributions are not normally distributed.


# In[58]:


sns.pairplot(df, hue='y')


# In[59]:


X = df.drop(['y', 'duration'], axis=1)
y = df['y']


# In[60]:


def make_mi_scores(x, y):

    x = x.copy()
    for colname in x.select_dtypes(["object", "category"]):
        x[colname], _ = x[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in x.dtypes]
    
    mi_scores = mutual_info_classif(x, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)


# In[61]:


mi_scores = make_mi_scores(X, y)
plot_mi_scores(mi_scores)


# In[62]:


# Our next step will be to transform our categorical variables in to the numeric.


# In[63]:


label_encoder = LabelEncoder()
for col in df.select_dtypes(["object", "category"]):
    df[col] = label_encoder.fit_transform(df[col])

df.head(15)


# In[64]:


# Some intermediate results
# For a now we have done several things:

# 1) We have checked our data, and found some missing values
# 2) We have cleaned up data
# 3) We have checked our target variable and found a problem.
# 4) We have checked the categorical variables and found that some groups are mispresenting in the data.


# In[65]:


# As a method of spliting the data set K-split was approved.
# First, lets calculate how many of the folds can be created from our data set.


# In[66]:


divisors = []
for i in range(1, df.shape[0] + 1):
    if df.shape[0] % i == 0:
        divisors.append(i)

results = [df.shape[0] / divisor for divisor in divisors]

print(results)


# In[67]:


# On the above, we see the number of the folds, which can be created. Let's create 7 folders, and save them


# In[68]:


import os

k_fold_splits = 7
kf = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
output_folder = '/Users/ilyashnip/Documents/GitHub/ML_project_in_Finance_I_Signing_of_deposit_agreement/ML_project_in_Finance_I_Signing_of_deposit_agreement/Data/Output'
os.makedirs(output_folder, exist_ok=True)
for i, (train_indices, test_indices) in enumerate(kf.split(df)):
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()

    train_filename = os.path.join(output_folder, f'df_train_fold_{i + 1}.csv')
    test_filename = os.path.join(output_folder, f'df_test_fold_{i + 1}.csv')

    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)

