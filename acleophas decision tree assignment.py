#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # Set to retina version")
pd.set_option('display.max_columns', None) # Set max columns output
warnings.filterwarnings('ignore')


# In[81]:


df = pd.read_csv("employee_attrition_dataset.csv")
print(df.shape)
display(df.head())


# # drop features with low variance or products of other features or with no significance in data analysis

# In[4]:


df = df.drop(columns=['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])


# In[8]:


#mapping and encoding categorical variables in your DataFrame


# In[5]:


education_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
education_satisfaction_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}
job_involvement_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}
job_satisfaction_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}
performance_rating_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
relationship_satisfaction_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}
work_life_balance_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
# Use the pandas apply method to numerically encode our attrition target variable
df['Education'] = df["Education"].apply(lambda x: education_map[x])
df['EnvironmentSatisfaction'] = df["EnvironmentSatisfaction"].apply(lambda x: education_satisfaction_map[x])
df['JobInvolvement'] = df["JobInvolvement"].apply(lambda x: job_involvement_map[x])
df['JobSatisfaction'] = df["JobSatisfaction"].apply(lambda x: job_satisfaction_map[x])
df['PerformanceRating'] = df["PerformanceRating"].apply(lambda x: performance_rating_map[x])
df['RelationshipSatisfaction'] = df["RelationshipSatisfaction"].apply(lambda x: relationship_satisfaction_map[x])
df['WorkLifeBalance'] = df["WorkLifeBalance"].apply(lambda x: work_life_balance_map[x])


# In[9]:


display(df.head())


# In[10]:


df.describe()


# In[11]:


df.Age.max()


# In[12]:


df.Age.min()


# In[13]:


df.Age.mode()


# In[14]:


df.Age.median()


# In[15]:


df.Age.mean()


# In[16]:


df.var()


# In[25]:


#extract specific column
df.Age


# # CHARTS

# In[84]:


sns.displot(df['Age'])


# In[85]:


sns.lineplot(df['Age'])


# In[86]:


sns.histplot(df['DailyRate'])


# In[87]:


sns.jointplot(df['Age'],kind='hex')


# In[88]:


sns.distplot(df['Age'])


# In[22]:


#sorting data by specific column
df.sort_values(by="Age")


# In[24]:


data=df.head(2)
data


# In[17]:


df.info()


# In[18]:


#Show all the columns in the dataframe
df.columns


# In[19]:


#data frmae size
df.shape


# # checjing for missing values

# In[23]:


#missing values
df.isnull().sum()


# In[20]:


print("Missing Value:", df.isnull().any().any())


# In[28]:


#checking for dupliacates in the dataset
print(df.duplicated().value_counts())
df.drop_duplicates(inplace=True)
print(len(df))


# # Distribution of  "Attrition"

# In[31]:


df.Attrition


# In[27]:


colors = ['#66b3ff', '#ff9999']
explode = (0.05,0.05)
plt.figure(figsize=(5, 5))
plt.pie(df['Attrition'].value_counts(), colors = colors, labels=['No', 'Yes'], 
        autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
plt.legend()
plt.title("Attrition (Target) Distribution")
plt.show()


# # Analysis of Numerical Features

# In[30]:


numerical_list = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                  'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

plt.figure(figsize=(10, 10))
for i, column in enumerate(numerical_list, 1):
    plt.subplot(5, 3, i)
    sns.distplot(df[column], bins=20)
plt.tight_layout()
plt.show()


# In[32]:


###Age: The age distribution of the dataset distributes normally which covers from 20 to 60. Most employees are 30 to 40.

###Most of employees live close to the company,most distance below 10km.

###The majority of monthly income of employees is around 5000. 

###Most employees have worked for one company.

###most people stay in the company for only for a few years.

###DailyRate, HourlyRate, and MonthlyRate are distributed uniformly which might imply that the figure is similar in different intervals.


# In[33]:


#Analysis of Categorical Features


# In[39]:


categorical_list = ['Attrition', 'BusinessTravel', 'Department', 'Education', 'EducationField', 
             'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole',
             'JobSatisfaction', 'MaritalStatus', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction',
             'StockOptionLevel', 'WorkLifeBalance']

# Create subplots
plt.figure(figsize=(20, 30))
gridspec = plt.GridSpec(7, 3)
locator1, locator2 = [0, 0]

for column in cate_list:
    plt.subplot(gridspec[locator1, locator2])
    sns.countplot(data=df, x=column, palette='Set2')
    plt.xticks(rotation=90)
    
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


# In[40]:


###most of employees are satisfied, of which the number of "3" and "4" indicate high satisfaction are a lot.
###Most employees hold bachelor degree 
###many people major in life science.
###most of employees are relatively new to the company
###job level 1 is the majority job level.


# # Correlation Analysis

# In[42]:


# Filter the DataFrame to include only numeric columns
numeric_columns = df.select_dtypes(include=[int, float])

# Create a correlation heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="Greys", annot_kws={"size": 15})
plt.show()


# In[43]:


###YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion and with YearsWithCurrManager are correlated

###TotalWorkingYears,JobLevel and MonthlyIncomeare correlated


# In[44]:


##Cross Analysis between Attrition and Numerical Features


# In[ ]:





# In[ ]:





# # Cross Analysis between Attrition and Categorical Features
# 

# In[49]:


# Create subplots
plt.figure(figsize=(20, 30))
gridspec = plt.GridSpec(7, 3)
locator1, locator2 = [0, 0]

for column in cate_list:
    if column == 'JobRole':
        plt.subplot2grid((7, 3), (locator1, locator2), colspan=3, rowspan=1)
        sns.countplot(x=column, hue='Attrition', data=df, palette='BrBG')
        locator1 += 1
        locator2 = 0
        continue
    plt.subplot2grid((7, 3), (locator1, locator2))
    sns.countplot(x=column, hue='Attrition', data=df, palette='BrBG')
    locator2 += 1
    if locator2 == 3:
        locator1 += 1
        locator2 = 0
        continue
    if locator1 == 7:
        break
plt.tight_layout()
plt.show()


# In[50]:


###Employees with business travel are more likely leave the company.

###Human Resource employees are the most stable group of employees.

###Employees with a Doctor's degree are stable.

###Technical employees tend to leave.

###Low performance rating and low stock option level may result employees' attrition


# # Feature selection 

# In[53]:


#Encoding of Categorical Features


# In[52]:


# Filter the list to exclude columns that don't exist in your DataFrame
cate_list_filtered = [col for col in cate_list if col in df.columns]

# Select the categorical columns from the DataFrame
data_categorical = df[cate_list_filtered]

# Display the head of the categorical data
data_categorical.head()


# In[89]:


#Encoding Categorical Values using OneHot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")
data_encoded = encoder.fit_transform(data_categorical)
data_encoded.head()


# In[56]:


#Encoding of Numerical Features


# In[57]:


###StandardScaler


# In[58]:


std = preprocessing.StandardScaler()
scaled = std.fit_transform(df[numerical_list])
scaled = pd.DataFrame(scaled, columns=numerical_list)

display(scaled.head())


# In[66]:


#Combine the encoded dataframe with the scaled dataframe
new_df = pd.concat([scaled, data_encoded], axis=1)
new_df.head()


# In[67]:


new_df.columns


# In[68]:


X = new_df[['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
       'Department_Sales', 'Education_Below College', 'Education_College',
       'Education_Doctor', 'Education_Master', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'EnvironmentSatisfaction_Low', 'EnvironmentSatisfaction_Medium',
       'EnvironmentSatisfaction_Very High', 'Gender_Male',
       'JobInvolvement_Low', 'JobInvolvement_Medium',
       'JobInvolvement_Very High', 'JobLevel_2', 'JobLevel_3', 'JobLevel_4',
       'JobLevel_5', 'JobRole_Human Resources',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Manufacturing Director', 'JobRole_Research Director',
       'JobRole_Research Scientist', 'JobRole_Sales Executive',
       'JobRole_Sales Representative', 'JobSatisfaction_Low',
       'JobSatisfaction_Medium', 'JobSatisfaction_Very High',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes',
       'PerformanceRating_Outstanding', 'RelationshipSatisfaction_Low',
       'RelationshipSatisfaction_Medium', 'RelationshipSatisfaction_Very High',
       'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
       'WorkLifeBalance_Best', 'WorkLifeBalance_Better',
       'WorkLifeBalance_Good']]

y = new_df['Attrition_Yes']


# In[73]:


#Getting feature importance using correlation

target = new_df['Attrition_Yes']

# Calculate Pearson correlation coefficients
correlations = new_df.drop(columns=['Attrition_Yes']).apply(lambda x: x.corr(target))

# Take absolute values and sort in descending order
correlations = correlations.abs().sort_values(ascending=False)

# Set the maximum column width to show the entire content
pd.set_option('display.max_colwidth', None)

# Print the ranked feature importances
with pd.option_context('display.max_rows', None):
    print(correlations)

# Reset the display options to their default values (if needed)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_colwidth')


# In[74]:


# Your correlations data
correlations = {
    'OverTime_Yes': 0.246118,
    'MaritalStatus_Single': 0.175419,
    'TotalWorkingYears': 0.171063,
    'YearsInCurrentRole': 0.160545,
    'MonthlyIncome': 0.159840,
    'Age': 0.159205,
    'JobRole_Sales Representative': 0.157234,
    'YearsWithCurrManager': 0.156199,
    'StockOptionLevel_1': 0.151049,
    'YearsAtCompany': 0.134392,
    'JobLevel_2': 0.131138,
    'EnvironmentSatisfaction_Low': 0.122819,
    'JobInvolvement_Low': 0.117161,
    'BusinessTravel_Travel_Frequently': 0.115143,
    'JobRole_Laboratory Technician': 0.098290,
    'MaritalStatus_Married': 0.090984,
    'JobSatisfaction_Low': 0.090329,
    'JobRole_Research Director': 0.088870,
    'JobSatisfaction_Very High': 0.087830,
    'JobLevel_4': 0.086461,
    'Department_Research & Development': 0.085293,
    'JobRole_Manager': 0.083316,
    'JobRole_Manufacturing Director': 0.082994,
    'Department_Sales': 0.080855,
    'StockOptionLevel_2': 0.080472,
    'DistanceFromHome': 0.077924,
    'EducationField_Technical Degree': 0.069355,
    'WorkLifeBalance_Better': 0.064301,
    'JobInvolvement_Very High': 0.063577,
    'TrainingTimesLastYear': 0.059478,
    'RelationshipSatisfaction_Low': 0.059222,
    'DailyRate': 0.056652,
    'EducationField_Marketing': 0.055781,
    'JobLevel_5': 0.053566,
    'BusinessTravel_Travel_Rarely': 0.049538,
    'EnvironmentSatisfaction_Very High': 0.047909,
    'EducationField_Medical': 0.046999,
    'JobInvolvement_Medium': 0.044731,
    'NumCompaniesWorked': 0.043494,
    'JobRole_Human Resources': 0.036215,
    'YearsSinceLastPromotion': 0.033019,
    'EducationField_Life Sciences': 0.032703,
    'Gender_Male': 0.029453,
    'Education_Doctor': 0.028507,
    'Education_Master': 0.025676,
    'RelationshipSatisfaction_Very High': 0.022940,
    'Education_Below College': 0.020777,
    'JobRole_Sales Executive': 0.019774,
    'EducationField_Other': 0.017898,
    'RelationshipSatisfaction_Medium': 0.017611,
    'JobLevel_3': 0.016380,
     'JobLevel_3': 0.016380,
    'EnvironmentSatisfaction_Medium': 0.015267,
    'MonthlyRate': 0.015170,
    'WorkLifeBalance_Best': 0.014131,
    'PercentSalaryHike': 0.013478,
    'WorkLifeBalance_Good': 0.011093,
    'StockOptionLevel_3': 0.010271,
    'Education_College': 0.006884,
    'HourlyRate': 0.006846,
    'JobSatisfaction_Medium': 0.004038,
    'PerformanceRating_Outstanding': 0.002889,
    'JobRole_Research Scientist': 0.000360,
}
# Initialize two empty lists
index = []
dropped = []

# Iterate through the correlations and categorize columns
for column, correlation in correlations.items():
    if correlation >= 0.05:
        index.append(column)
    else:
        dropped.append(column)

# Print the lists
print("Selected featured:", index)
print("Dropped features:", dropped)


# In[75]:


new_df = new_df.drop(columns=dropped)
new_df.columns


# In[76]:


#Model Building


# In[83]:


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# machine learning model
model = DecisionTreeClassifier()

# Load your dataset and perform feature/target splitting
X = new_df.drop(columns=['Attrition_Yes'])
y = new_df['Attrition_Yes']

# Define the number of folds
num_folds = 5

# Create a K-Fold cross-validator with 5 folds
kf = KFold(n_splits=num_folds)

# Lists to store cross-validation results
train_scores = []
test_scores = []

# Perform k-fold cross-validation and calculate performance metrics
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training fold
    model.fit(X_train_fold, y_train_fold)

    # Make predictions on the test fold
    y_test_pred = model.predict(X_test_fold)

    # Calculate accuracy on the training and test folds
    train_accuracy = accuracy_score(y_train_fold, model.predict(X_train_fold))
    test_accuracy = accuracy_score(y_test_fold, y_test_pred)

    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

# Print the cross-validation results
for fold, (train_score, test_score) in enumerate(zip(train_scores, test_scores), start=1):
    print(f"Fold {fold}: Train Accuracy = {train_score:.2f}, Test Accuracy = {test_score:.2f}")

mean_train_accuracy = sum(train_scores) / len(train_scores)
mean_test_accuracy = sum(test_scores) / len(test_scores)

print(f"Mean Train Accuracy = {mean_train_accuracy:.2f}")
print(f"Mean Test Accuracy = {mean_test_accuracy:.2f}")


# In[90]:


from sklearn.metrics import confusion_matrix

# ... (Previous code for k-fold cross-validation)

# Lists to store confusion matrices
confusion_matrices = []

# Perform k-fold cross-validation and calculate confusion matrices
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training fold
    model.fit(X_train_fold, y_train_fold)

    # Make predictions on the test fold
    y_test_pred = model.predict(X_test_fold)

    # Calculate the confusion matrix for the test fold
    confusion_matrix_fold = confusion_matrix(y_test_fold, y_test_pred)
    confusion_matrices.append(confusion_matrix_fold)

# Print the confusion matrices for each fold
for fold, confusion_matrix_fold in enumerate(confusion_matrices, start=1):
    print(f"Confusion Matrix for Fold {fold}:\n{confusion_matrix_fold}")


# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ... (Previous code for k-fold cross-validation)

# Lists to store confusion matrices and labels
confusion_matrices = []
fold_labels = []

# Perform k-fold cross-validation and calculate confusion matrices
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training fold
    model.fit(X_train_fold, y_train_fold)

    # Make predictions on the test fold
    y_test_pred = model.predict(X_test_fold)

    # Calculate the confusion matrix for the test fold
    confusion_matrix_fold = confusion_matrix(y_test_fold, y_test_pred)
    confusion_matrices.append(confusion_matrix_fold)
    fold_labels.append(f"Fold {fold}")

# Plot confusion matrices
for fold, confusion_matrix_fold in enumerate(confusion_matrices):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix_fold, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix - {fold_labels[fold]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[ ]:




