#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import re
from joblib import dump

# Load your dataset
df = pd.read_csv('S:\ML Course\Midterm Project 1\mobile_price.csv')

# Drop the 'Unnamed: 8' column that was showing up
if 'Unnamed: 8' in df.columns:
    df = df.drop('Unnamed: 8', axis=1)


    
len(df)


# In[96]:


df.head()


# In[97]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
df.head()


# In[98]:


# Rename column price_($) to price
df.rename(columns={'price_($)':'price'}, inplace=True)

# Convert 'price' to numeric and handle any non-numeric entries.
df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)

# Handle NaN values that may have been introduced by non-numeric entries.
df['price'].fillna(df['price'].median(), inplace=True)

# Now you can safely log transform the price column.
df['log_price'] = np.log1p(df['price'])


plt.figure(figsize=(8, 6))

sns.histplot(df['price'], bins=50, color='black', alpha=1)
plt.ylabel('Quantity')
plt.xlabel('Price')
plt.title('Distribution of prices')

plt.show()


# In[99]:


df.isnull().sum()


# In[106]:


# Validation Framework
np.random.seed(2)
n = len(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

y_train_orig = df_train.price.values
y_val_orig = df_val.price.values
y_test_orig = df_test.price.values

y_train = np.log1p(y_train_orig)
y_val = np.log1p(y_val_orig)
y_test = np.log1p(y_test_orig)

del df_train['price']
del df_val['price']
del df_test['price']

# Linear Regression Model Training
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Feature Preparation
base = ['storage', 'screen_size_(inches)', 'camera_(mp)']  # Make sure these are the correct feature names

def prepare_X(df):
    df_num = df.copy()
    for col in base:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce')
    df_num = df_num.fillna(0)
    X = df_num[base].values
    return X

# Prepare the data
X_train = prepare_X(df_train)
X_val = prepare_X(df_val)

# Train the model
model = train_linear_regression(X_train, y_train)

# Save the trained model
dump(model, 'model.joblib')

# Make predictions
y_pred = model.predict(X_val)

# Plot the predictions vs actual distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_val, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')
plt.show()


# In[115]:


from flask import Flask, request, jsonify
from joblib import load
from threading import Thread

app = Flask(__name__)

# Load the trained model
model = load('model.joblib')

@app.route('/')
def home():
    return "Welcome to the model prediction service!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the input data as required, similar to how you did in the notebook
    # For example, if you expect a single feature called 'feature_input'
    input_data = [data['feature_input']]
    # Use the model to make a prediction
    prediction = model.predict([input_data])
    return jsonify({'prediction': prediction.tolist()})

# Define the function that will run the Flask app
def run_app():
    # Set the threaded argument to True to handle each request in a separate thread.
    app.run(port=6969, debug=True, use_reloader=False, threaded=True)

# Run the Flask app in a separate thread to avoid blocking the notebook
flask_thread = Thread(target=run_app)
flask_thread.start()


# In[116]:


pip freeze > requirements.txt


# In[ ]:




