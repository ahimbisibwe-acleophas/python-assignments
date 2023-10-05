#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import matplotlib.gridspec as gridspec
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from IPython.display import Image
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # Set to retina version")
pd.set_option('display.max_columns', None) # Set max columns output
warnings.filterwarnings('ignore')


# In[78]:


#IMPORT CSV FILE
df=pd.read_csv('employee_attrition_dataset.csv')
df


# # dropping features with very low variance and with similar values

# In[23]:


df = df.drop(['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'], axis=1)


# In[79]:


df.describe()


# In[12]:


df.Age.max()


# In[13]:


df.Age.min()


# In[14]:


df.Age.mode()


# In[15]:


df.Age.median()


# In[16]:


df.Age.mean()


# In[53]:


df.var()


# In[26]:


df.info()


# # column headings

# In[57]:


df.columns


# In[59]:


#data frmae size
df.shape


# In[54]:


#sorting data by specific column
df.sort_values(by="Age")


# In[106]:


data=df.head(2)
data


# In[107]:


data=df.tail(2)
data


# In[59]:


#extract specific column
df.Age


# In[97]:


#missing values
df.isnull().sum()


# In[98]:


#checking for dupliacates in the dataset
print(df.duplicated().value_counts())
df.drop_duplicates(inplace=True)
print(len(df))


# In[99]:


print(df.duplicated().value_counts())


# In[149]:


df.Attrition


# In[151]:


# Explore the distribution of Attrition
data=df['Attrition'].value_counts()
data


# In[152]:


# Explore the distribution of EducationField
df['EducationField'].value_counts()


# In[28]:


df=df.set_index('Attrition')
display(df)
df.info()


# In[ ]:





# # Visualize using Age column

# In[60]:


df[['Department', 'Age']]


# In[61]:


df['DailyRate'].cumsum()


# In[ ]:





# # CHARTS

# In[62]:


sns.displot(df['Age'])


# In[63]:


sns.lineplot(df['Age'])


# In[64]:


sns.histplot(df['DailyRate'])


# In[65]:


sns.jointplot(df['Age'],kind='hex')


# In[66]:


sns.stripplot(df['Age'])


# In[67]:


sns.distplot(df['Age'])


# In[68]:


plt.figure(figsize=(12,5))
sns.countplot(x='Gender',hue='Attrition',data=df, palette='hot')
plt.title( 'Attrition VS Gender')
plt.legend(loc='best')
plt.show()


# # distribution of some numerical columns in a df

# In[126]:


Numerical_list = ['DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                  'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_list, 1):
    plt.subplot(5, 3, i)
    sns.distplot(df[column], bins=20)
plt.tight_layout()
plt.show()


# # distribution of some numerical data

# In[122]:


categorical_list = ['BusinessTravel', 'Department', 'Education', 'EducationField', 
             'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole',
             'JobSatisfaction', 'MaritalStatus', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction',
             'StockOptionLevel', 'WorkLifeBalance']

# Create subplots
plt.figure(figsize=(15, 25))
gridspec = plt.GridSpec(7, 3)
locator1, locator2 = [0, 0]

for column in cate_list:
    plt.subplot(gridspec[locator1, locator2])
    sns.countplot(data=df, x=column, palette='Set2')
    plt.xticks(rotation=30)
    
    locator2 += 1
    if locator2 == 3:
        locator1 += 1
        locator2 = 0
        continue
    if locator1 == 7:
        break

    # Set the y-axis limits to display more values
    plt.ylim(0, len(df))  # You can adjust the range as needed

plt.tight_layout()
plt.show()


# In[ ]:


### most of employees are satisfied, of which the number of "3" and "4" indicate high satisfaction are a lot.

Most employees hold bachelor degree
many people major in life science.
job level 1 is the majority job level.



# In[ ]:





# # ENCODING CATEGORICAL DATA

# In[68]:


convert={'Attrition':{"Yes":1, "No":0}}
df2=df.replace(convert)
df2
convert={'Department':{"Sales":1, "Research & Development":0}}
df2=df.replace(convert)
df2
convert={'EducationField':{"Medical":2, "Life Sciences":1, "Other":0}}
df=df.replace(convert)
df2
convert={'BusinessTravel':{"Non_Travel":2, "Travel_Frequently":1, "Travel_Rarely":0}}
df=df.replace(convert)
df2
convert={'Gender':{"Male":1, "Female":0}}
df2=df.replace(convert)
df2
convert={'JobRole':{"Manager":6,"Sales Representative":5,"Healthcare Representative":4,"Manufacturing Director":3,"Laboratory Technician":2,"Sales Executive":1, "Research Scientist":0}}
df2=df.replace(convert)
df2
convert={'Over18':{"Y":1}}
df2=df.replace(convert)
df2
convert={'OverTime':{"Yes":1,"No":0}}
df2=df.replace(convert)
df2
convert={'MaritalStatus':{"Divorced":3,"Married":1, "Single":0}}
df2=df.replace(convert)
df2


# # frquency tables

# In[79]:


#create a frequency table from attribute attrition
freq_table = pd.crosstab(df['Attrition'], 'no_of_attirition')
freq_table


# In[80]:


freq_table = pd.crosstab(df['EducationField'], 'no_of_attirition')
freq_table


# In[153]:


sorted_df=df.sort_values(by='Age',ascending=True)
sorted=sorted_df['Age']
sorted


# In[154]:


freq_table = pd.crosstab(index=[df['Attrition'], df['Department']], columns=df['Education'])
freq_table


# In[ ]:





# In[ ]:





# # FEATURE SELECTION

# In[40]:


#CORRELATION
# Calculate the correlation matrix between features
# Filter the DataFrame to include only numeric columns
numeric_columns = df.select_dtypes(include=[int, float])

# Create a correlation heatmap
plt.figure(figsize=(25, 25))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="Greys", annot_kws={"size": 15})
plt.show()


# ### YearsAtCompany, yearsInCurrentRole, YearswithCurentManager and Yearssincelast promotion are correlated
# TotalWorkingYears,MonthlyIncome and JobLevel are correlated 
# 

# In[91]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[93]:


from sklearn.ensemble import RandomForestClassifier

# Separate features and target variable
X = df2.drop('Attrition', axis=1)
y = df2['Attrition']


# In[95]:


df2 = pd.DataFrame({'Attrition': [0, 1]})
df3 =df2.rename(columns={'Attrition_Yes': 'Attrition'})


# In[97]:


# Separate features and target variable
X = df3.drop('Attrition', axis=1)
y = df3['Attrition']


# In[98]:


df.columns


# In[ ]:





# # cross analysis of categorical and numerical data
# cross validation 
# purpose

# In[ ]:





# In[ ]:





# In[174]:


# Iterate through categorical features
for categorical_feature in df.select_dtypes(include='object').columns:
    le = preprocessing.LabelEncoder()
    df[cate_feature] = le.fit_transform(df[cate_feature])

# Now, you can display the encoded DataFrame
df.head()


# In[ ]:





# In[ ]:





# In[99]:


dummies = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(data=df, columns=dummies)
display(df.head())


# In[ ]:





# In[59]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score


# In[60]:


from sklearn.model_selection import train_test_split
x=df.drop('Attrition', axis=1)
y=df['Attrition']

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=42)

print("shape of x_train:", x_train.shape)
print("shape of x_test:", x_test.shape)
print("shape of y_train:", y_train.shape)
print("shape of y_test:", y_test.shape)


# In[ ]:





# In[110]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier  # Replace with your model
from sklearn.datasets import load_iris  # Replace with your dataset

# Load your dataset (replace with your own dataset)
X, y = load_iris(return_X_y=True)

# Create a model (replace with your own model)
model = RandomForestClassifier()

# Specify the number of folds (2 in this case)
num_folds = 2

# Create a K-Fold cross-validator with 2 folds
kf = KFold(n_splits=num_folds)

# Perform 2-fold cross-validation and calculate accuracy scores
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Print the accuracy scores for each fold and their mean
for fold, score in enumerate(scores, start=1):
    print(f"Fold {fold}: Accuracy = {score:.2f}")

mean_accuracy = scores.mean()
print(f"Mean Accuracy = {mean_accuracy:.2f}")



# In[ ]:


interpretatation of mean accuracy = 0.32


# In[ ]:




