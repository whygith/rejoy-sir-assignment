#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


# Load the dataset (Assume 'drugdataset.csv' is in the current working directory)
file_path = "drugdataset.csv"  # Replace with the correct path
data = pd.read_csv(r"C:\Users\Sarthak\Downloads\drugdataset (1).csv")


# In[4]:


# Display first few rows of the dataset
print("Dataset Preview:")
print(data.head())


# In[5]:


# Encoding categorical variables (if necessary)
label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


# In[6]:


# Independent and Dependent variables
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']


# In[7]:


# Splitting the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[8]:


# Standardizing the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


# Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(5, 4, 5), max_iter=10000, random_state=100)
nn_model.fit(X_train, y_train)


# In[10]:


# Predictions and Evaluations
y_pred = nn_model.predict(X_test)


# In[11]:


# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[12]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[13]:


# Save classification report as HTML for submission
classification_report_html = classification_report(y_test, y_pred, output_dict=True)
classification_report_df = pd.DataFrame(classification_report_html).transpose()
classification_report_df.to_html("classification_report.html")


# In[ ]:




