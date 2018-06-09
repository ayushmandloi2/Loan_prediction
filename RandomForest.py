
# coding: utf-8

# In[168]:


import pandas as pd
import numpy as np


# In[387]:


train = pd.read_csv('G:/ml/loan_prediction/train.csv')
test = pd.read_csv('G:/ml/loan_prediction/test.csv')


# In[388]:


len(train[train['Loan_Status']== 'Y'])


# In[389]:


train['Loan_Status']=train['Loan_Status'].apply(lambda x: 1 if x =='Y' else 0)


# In[390]:


print(100*np.mean(train['Loan_Status'][train['Gender'] =='Male']))
print(100*np.mean(train['Loan_Status'][train['Gender'] =='Female']))


# In[391]:


print(100*np.mean(train['Loan_Status'][train['Married'] =='Yes']))
print(100*np.mean(train['Loan_Status'][train['Married'] =='No']))


# In[392]:


print(100*np.mean(train['Loan_Status'][train['Education'] =='Graduate']))
print(100*np.mean(train['Loan_Status'][train['Education'] =='Not Graduate']))


# In[393]:


print(100*np.mean(train['Loan_Status'][train['Self_Employed'] =='Yes']))
print(100*np.mean(train['Loan_Status'][train['Self_Employed'] =='No']))


# In[394]:


print(100*np.mean(train['Loan_Status'][train['Property_Area'] =='Urban']))
print(100*np.mean(train['Loan_Status'][train['Property_Area'] =='Rural']))
print(100*np.mean(train['Loan_Status'][train['Property_Area'] =='Semiurban']))


# In[395]:


print(pd.isnull(test).sum())


# In[396]:


train['Gender']=train['Gender'].apply(lambda x: 1 if x =='Male' else 0)
train['Married']=train['Married'].apply(lambda x: 1 if x =='Yes' else 0)
train['Education']=train['Education'].apply(lambda x: 1 if x =='Graduate' else 0)
train['Self_Employed']=train['Self_Employed'].apply(lambda x: 1 if x =='Yes' else 0)
train['Property_Area']=train['Property_Area'].apply(lambda x: 1 if x =='Semiurban' else 0)
test['Gender']=test['Gender'].apply(lambda x: 1 if x =='Male' else 0)
test['Married']=test['Married'].apply(lambda x: 1 if x =='Yes' else 0)
test['Education']=test['Education'].apply(lambda x: 1 if x =='Graduate' else 0)
test['Self_Employed']=test['Self_Employed'].apply(lambda x: 1 if x =='Yes' else 0)
test['Property_Area']=test['Property_Area'].apply(lambda x: 1 if x =='Semiurban' else 0)


# In[397]:



from sklearn.preprocessing import LabelEncoder
var_mod= ['Dependents']
le = LabelEncoder()
train['Dependents'] =le.fit_transform(train['Dependents'])
test['Dependents'] =le.fit_transform(test['Dependents'])


# In[398]:


Z=test.drop(['Loan_ID'],axis=1)
M = test['Loan_ID']


# In[399]:


train['Gender']=train['Gender'].fillna(np.mean(train['Gender']))
train['Married']=train['Married'].fillna(np.mean(train['Married']))
train['Dependents']=train['Dependents'].fillna(np.mean(train['Dependents']))
train['Self_Employed']=train['Self_Employed'].fillna(np.mean(train['Self_Employed']))
train['LoanAmount']=train['LoanAmount'].fillna(np.mean(train['LoanAmount']))
train['Loan_Amount_Term']=train['Loan_Amount_Term'].fillna(np.mean(train['Loan_Amount_Term']))
train['Credit_History']=train['Credit_History'].fillna(np.mean(train['Credit_History']))
train =train[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']]
test =test[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]


# In[400]:


test['Dependents']=test['Dependents'].fillna(np.mean(test['Dependents']))
test['LoanAmount']=test['LoanAmount'].fillna(np.mean(test['LoanAmount']))
test['Loan_Amount_Term']=test['Loan_Amount_Term'].fillna(np.mean(test['Loan_Amount_Term']))
test['Credit_History']=test['Credit_History'].fillna(np.mean(test['Credit_History']))


# In[401]:



X = train.drop(['Loan_Status'],axis =1)

y = train['Loan_Status']
train.describe()


# In[402]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state =42, test_size =0.3)
print(X_test.head())


# In[403]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators =50,max_depth =5)
clf.fit(X_train,y_train)


# In[404]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,clf.predict(X_train)))
print(accuracy_score(y_test,clf.predict(X_test)))


# In[405]:


pred = clf.predict(test)
print(pred)


# In[407]:


submission = pd.DataFrame({'Loan_ID': M,'Loan_Status': pred})
submission.to_csv('G:/ml/loan_prediction/submission.csv')

