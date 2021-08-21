#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda --version


# ### Overview
# The data has been split into two groups:
# * training set (train.csv)
# * test set (test.csv)
# The training is set used to build The machine learning models. 
# 
# The test set is used to see how well The model performs on unseen data. For the test set, We will predict if this Passenger is Survived or Sinked
# 
# The gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

# ### Data Describe : -
# #### survival  ====> Survival  
#  * 0 = No
#  * 1 = Yes
# #### pclass    ====> Ticket class  
#  * 1 = 1st
#  * 2 = 2nd
#  * 3 = 3rd
# #### sex       ====> Sex
# #### Age       ====> Age in years
# #### sibsp     ====> # of siblings / spouses aboard the Titanic	
# #### parch     ====> # of parents / children aboard the Titanic	
# #### ticket    ====> Ticket number
# #### fare      ====> Passenger fare
# #### cabin     ====> Cabin number
# #### embarked  ====> Port of Embarkation
# * C = Cherbourg 
# * Q = Queenstown
# * S = Southampton
# 

# In[2]:


# Importing the Important Library :

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[3]:


# Importing The Training Dataset :

train_data = pd.read_csv("data/train.csv")
train_data.head()


# In[4]:


train_data.shape


# In[5]:


# general view about missing data on Training Dataset:

train_data.isna().sum()


# In[6]:


# Fill The missing training data on Age with mean :

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
train_data['Age'] = imp.fit_transform(np.array(train_data['Age']).reshape(-1,1)).astype(int)


# In[7]:


train_data.isna().sum()


# In[8]:


# drop unnecessary Columns :

train_data = train_data.drop(['PassengerId','Name' , 'Fare' , 'Ticket' , "Cabin"], axis=1)
train_data.head()


# In[9]:


train_data.shape


# In[10]:


# Convert string value with numerical :

from sklearn.preprocessing import LabelEncoder 
en = LabelEncoder()
train_data['Sex'] = en.fit_transform(train_data['Sex'])
train_data['Embarked'] = en.fit_transform(train_data['Embarked'])
train_data.head()


# In[11]:


# Split the Training Dataset to  Features and Target :

X_train = train_data.drop('Survived' , axis = 1)
y_train = train_data.Survived

print(f" The Shape of X_train = {X_train.shape}\n The shape of y_train = {y_train.shape} ")


# In[12]:


train_data.head()


# In[13]:


sns.set_theme = "ticks"
sns.barplot(x = y_train , y = train_data["Age"])
plt.show()


# In[14]:


# Importing The Test Dataset :

test_data = pd.read_csv("data/test.csv")
test_data.head()


# In[15]:


# general view about missing data on Testing Dataset :

test_data.isna().sum()


# In[16]:


# Fill The missing testing data on Age with mean :

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
test_data['Age'] = imp.fit_transform(np.array(test_data["Age"]).reshape(-1,1)).astype(int)


# In[17]:


test_data.isna().sum()


# In[18]:


# drop unnecessary Columns :

test_data = test_data.drop(['PassengerId','Name' , 'Fare' , 'Ticket' , "Cabin"], axis=1)
test_data.head()


# In[19]:


# Convert string value with numerical :

from sklearn.preprocessing import LabelEncoder 
en = LabelEncoder()
test_data['Sex'] = en.fit_transform(test_data['Sex'])
test_data['Embarked'] = en.fit_transform(test_data['Embarked'])
test_data.head()


# In[20]:


# Importing The Anothe Testing Dataset :

test_data2 = pd.read_csv('data/gender_submission.csv')
test_data2.head()


# In[21]:


test_data2.isna().sum()


# In[22]:


# Split the Testing Dataset to  Features and Target :

X_test = test_data
y_test = test_data2['Survived']


# In[23]:


print(f"""
At The final :\n
The Shape of X_train : {X_train.shape} .
The Shape of y_train : {y_train.shape} .
The Shape of X_test  : {X_test.shape} .
The Shape of y_test  : {y_test.shape} .
""")


# In[24]:


# It is a Collection of Machine Learning Algorithms To Estimate And Select The Best Model. 

# Classification : 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

models = {'Logestic_Regression' : LogisticRegression() ,
          'KNN' : KNeighborsClassifier() ,
          'Random_Forest_Classifier' : RandomForestClassifier() ,
          'SVC' : SVC() ,
          'Decision_Tree' : DecisionTreeClassifier()
          }

def fit_and_score(models , X_train , X_test , y_train , y_test) :
    model_scores = {}
    model_confusion = {}
    for name , model in models.items() :
        # fitting the data :
        model.fit(X_train , y_train)
        model_scores[name] = model.score(X_test , y_test)
        y_predict = model.predict(X_test)
        model_confusion[name] = confusion_matrix(y_test , y_predict)
    return model_scores , model_confusion


# In[25]:


# Calling the Function :

fit_and_score(models = models ,
              X_train = X_train,X_test = X_test,
              y_train = y_train,y_test = y_test )


# ### Great .....
# #### The Best Machine Learning Algorithm With a Best Accuracy : Logestic_Regression .

# In[26]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

lr = LogisticRegression()
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test , y_pred)
heat = sns.heatmap(cm)
print(f"""
The Score Of Model   : {lr.score(X_test , y_test)} .
The accuracy Score   : {accuracy_score(y_test , y_pred)} .
The Classification Report \n {classification_report(y_test , y_pred)}
""")
print(f"The Confusion Matrix : {cm}")
plt.show()


# In[ ]:




