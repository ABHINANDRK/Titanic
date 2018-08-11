
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# In[2]:


os.chdir("/home/rk/Desktop/kaggle/all")


# In[3]:


trainSample=pd.read_csv("train.csv")

testSample=pd.read_csv("test.csv")

titanic =  pd.concat(objs=[trainSample,testSample],axis=0).reset_index(drop=True)


# In[4]:


trainSample['Died']=1-trainSample['Survived']
trainSample.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, colors=['g', 'r'])


# In[5]:



missing_val=pd.DataFrame(titanic.isnull().sum())
missing_val=missing_val.reset_index()
missing_val=missing_val.rename(columns={'indrex':'variables',0:'missing_percentage'})
rmissing_val['missing_percentage']=((missing_val['missing_percentage'])/len(titanic))*100
missing_val


# In[6]:


titanic=titanic[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]


# In[7]:


titanic=titanic.drop(['Cabin'],axis=1)


# In[8]:


#Creating new variable for 'major','sir'...so that we will get more information about our dataset.there by we can increase our model accuracy 
#me:nice idea though :)

titles = set()
for name in titanic['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())


# In[9]:


Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}


# In[11]:


# we extract the title from each name
titanic['Title'] = titanic['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
  
  # a map of more aggregated title
  # we map each title
titanic['Title'] = titanic.Title.map(Title_Dictionary)
#  status('Title')
 # return combined


# In[12]:


titanic.columns


# In[13]:


list=[]

for i in range(0,titanic.shape[1]):
    if(titanic.iloc[:,i].dtypes == 'object'):
        
        titanic.iloc[:,i]=pd.Categorical(titanic.iloc[:,i])
        titanic.iloc[:,i]=titanic.iloc[:,i].cat.codes
        list.append(titanic.columns[i])
    
    


# In[14]:


for i in range(0,titanic.shape[1]):
    titanic.iloc[:,i]=titanic.iloc[:,i].replace(-1,np.nan)


# In[ ]:


titanic=pd.DataFrame(KNN(k=3).complete(titanic),columns=titanic.columns)


# In[16]:


for i in list:
    titanic.loc[:,i]=titanic.loc[:,i].round()
    titanic.loc[:,i]=titanic.loc[:,i].astype('object')


# In[17]:


cnames=["PassengerId","Pclass","Age","SibSp","Parch","Fare"]
#save categorical values
cat_names=["Name","Sex","Ticket","Embarked","Title"]


# In[18]:


#feature selection
dff_corr=titanic.loc[:,cnames]

f,ax=plt.subplots(figsize=(10,8))

corr=dff_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[ ]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(titanic['Survived'], titanic[i]))
    print(p)


# In[20]:


titanic=titanic.drop(["Name"],axis=1)


# In[ ]:


#normality check.can check normality of each variable 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(titanic['Age'],bins='auto')


# In[22]:


#im not sure about this line try standardization also :)
for i in cnames:
    titanic[i]=(titanic[i]-np.min(titanic[i]))/(np.max(titanic[i])-np.min(titanic[i]))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(titanic['Age'],bins='auto')


# In[24]:


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[ ]:


for i in range(890,892):
    print(titanic['Survived'].iloc[i])r


# In[26]:


#model developement


#replace target categories with Yes or No
titanic['Survived'] = titanic['Survived'].replace(0, 'No')
titanic['Survived'] = titanic['Survived'].replace(1, 'Yes')


# In[27]:


titanic=titanic.drop(["PassengerId"],axis=1)


# In[ ]:


#train=titanic.head(892)
#test=titanic.tail(417)
titanic.dtypes


# In[28]:


titanic.to_csv("abc.csv",index=False)


# In[ ]:


titanic.columns


# In[46]:


titanic=titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked','Title','Survived']]


# In[47]:


train=titanic.iloc[0:892,]


train['Survived'].iloc[891]='No'


# In[ ]:


test=titanic.iloc[891:1309,]
test.describe()


# In[49]:


train.tail(10)


# In[50]:


#Divide data into train and test



X = train.values[:,0:9]
Y = train.values[:,9]

Xt = test.values[:,0:9]
#Yt = test.values[:,8]

#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)
#X_train, X_test, y_train, y_test = train_test_split( X, Y)


# In[51]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X,Y)


# In[ ]:


RF_Predictions = RF_model.predict(Xt)
ans=pd.DataFrame({'PassengerId':range(892,892+len(RF_Predictions)),'Survived':RF_Predictions})
ans


# In[53]:


#ans=ans["Survived"].replace("Yes",1)
#ans=ans["Survived"].replace("No",0)
ans['Survived'] = ans['Survived'].map({'Yes': 1, 'No': 0})
ans.to_csv("submission.csv",index=False)


# In[566]:


#ans.to_csv("newsubmission.csv",index=False)


# In[263]:



#Since we don't have test result we cant't find the accuracy of the model.so keep calm ,upload the result to kaggle :)

#CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
#TN = CM.iloc[0,0]
#FN = CM.iloc[1,0]
#TP = CM.iloc[1,1]
#FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)

