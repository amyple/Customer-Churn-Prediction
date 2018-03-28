
# coding: utf-8

# In[209]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[210]:


df = pd.read_csv("C:/Users/Amy Le - PC/Documents/WA_Fn-UseC_-Telco-Customer-Churn.csv", low_memory = False)
df = df.replace(" ", np.NaN)
###The telecom dataset is collected from IBM Watson Analytics 
###"https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/"


# In[211]:


df.info()


# In[212]:


print(df.shape)
df.head()
### The dataset has 21 attributes and 7043 instances(customers)
### Churn (yes, no) is the target feature


# Data Preprocessing
# 

# In[213]:


df.isnull().sum()
### Checking missing values-> there are 11 missing values in “TotalCharges” column.


# In[214]:


df.dropna(inplace=True) #remove all rows with missing values


# In[215]:


### I assume that "No internet service" value is the same as "No" value for the following attributes: MultipleLines; 
#OnlineSecurity; OnlineBackup; DeviceProtection; TechSupport; StreamingTV; StreamingMovies
### "No phone service" value is the same as "No" value for "MultipleLine" attribute

df.replace("No internet service", "No", inplace = True)
df.replace("No phone service", "No", inplace = True)

### Change the value for 'SeniorCitizen' attribute
replacements = {
  1: 'Yes',
  0: 'No'
}
df['SeniorCitizen'].replace(replacements, inplace=True)

#Remove 'Customer_ID' column
df = df.drop(['customerID'], axis =1)

# Convert'TotalCharges' to numeric type
df['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric) 




# In[216]:


#Correlation Between continuous attributes
correlations = df.corr(method='pearson')
print(correlations)
sns.heatmap(correlations)
### As we can see, 'TotalCharges' are correlated with both 'tenure' and 'MonthlyCharges', so I remove this attribute 
#from the model.


# In[217]:


df.drop(['TotalCharges'], axis =1, inplace = True)


# In[218]:


### Bar plots of categorical value
cat_col = df.select_dtypes(exclude=[np.number]).columns.tolist()
for feature in cat_col:
   sns.factorplot(feature,data=df,kind='count')


# In[219]:


### The categorical variables seem to have broad distribution, so I keep them for analysis


# In[220]:


###Label encoding
#Creating a list of categorical features

for i in df.columns:
    if i in cat_col:
        df[i] = df[i].astype('category').cat.codes
df.head()


# In[221]:


#Class distribution
class_counts = df.groupby('Churn').size()
print(class_counts)

#Outcome of this dataset is a majority of 'Churn ==0 (no)' class. Therefore, F1 score is more useful than accuracy score 
#as a metric for model evaluation. 


# In[222]:


Y = df.iloc[:,-1]


# In[223]:


X = df.drop(['Churn'], axis =1)


# In[224]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics


X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, X.index, test_size=0.3, random_state=0)

###Make sure the frequency of churn is approximately the same for training data as well as for test data

print('% Churn: ', Y[Y== 1].shape[0] / Y.shape[0] * 100)
print('Train - size:', y_train.shape[0], ', %Churn:', y_train[y_train == 1].shape[0] / y_train.shape[0] * 100)
print('Test - size:', y_test.shape[0], ', %Churn: ', y_test[y_test == 1].shape[0] / y_test.shape[0] * 100)


# In[225]:


# Suport Vector Machine classifier
model1 = LinearSVC()
# Train the supervised model on the training set using .fit(X_train, y_train)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test) #predicted class
print(metrics.classification_report(y_test, y_pred1))


# In[226]:


#Random Forest classifier

model2 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# Train the supervised model on the training set using .fit(X_train, y_train)
model2 = model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(metrics.classification_report(y_test, y_pred2))



# In[227]:


# Extract the feature importances using .feature_importances_ 
importances = model2.feature_importances_
print(X_train.columns)
print(importances)


# In[228]:


# Print the feature ranking
std = np.std([tree.feature_importances_ for tree in model2.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()



# From the graph, we learn that the 'MonthlyCharges', 'tenure','Contract', and 'Payment Method'  are the key features that has the most importance on the whether or not customers will leave.

# In[229]:


#Logistic Regression
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg.coef_)
y_pred3 = logreg.predict(X_test)
logreg.score(X_test, y_test) 
print(metrics.classification_report(y_test, y_pred3))



# In[230]:


g = sns.PairGrid(df,
                 y_vars= ['tenure'],
                 x_vars=['Churn'],
                 aspect=1, size=3.5)
g.map(sns.barplot, palette="pastel");


# In[231]:


g = sns.PairGrid(df,
                 y_vars= ['MonthlyCharges'],
                 x_vars=['Churn'],
                 aspect=1, size=3.5)
g.map(sns.barplot, palette="pastel");


# In[232]:


g = sns.PairGrid(df,
                 y_vars= ['tenure'],
                 x_vars=['Churn'],
                 aspect=1, size=3.5)
g.map(sns.barplot, palette="pastel");


# In[233]:


pd.crosstab(df.Contract,df.Churn).plot(kind='bar')
plt.xlabel('Contract')
plt.ylabel('Churn')


# In[235]:


pd.crosstab(df.PaymentMethod,df.Churn).plot(kind='bar')
plt.xlabel('PaymentMethod')
plt.ylabel('Churn')


# Conclusion
# 

# When a customer leaves, how often does the classifier predict that correctly? 
# -> “Recall” and a quick look at these diagrams can demonstrate that SVM is best for this criteria with 79%
# 
# When a classifier predicts a customer will leave, how often does that customer actually leave? 
# -> “Precision” of Logistic Regression with 64%
# 
# F1-score of Logistic Regression with 79%
# 

# Some important things about the features:
# 
# -tenure, Contract, PaymentMethod, MonthlyCharges play a role in customer churn.
# -Customers use electronic check; on a month-to-month contract; within 12 months tenure, are more likely to churn.
# -On the other hand, customers with one or two year contract; longer than 12 months tenure are less likely to churn.
