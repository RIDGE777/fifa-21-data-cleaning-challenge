#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning Challenge 2

# After exporting the first FIFA 21 Data Cleaning Challenge fifa_clean dataset, I realized that the special characters are still an issue in the csv file. I have to clean the fifa_clean dataset and set the correct datatype for each column
# 

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime 
import matplotlib.pyplot as plt


# In[25]:


file_location = 'C:\\Users\\ragir\\OneDrive\\Desktop\\DATA ANALYTICS\\Portfolio Projects\\Data Cleaning Challenge\\'

fifa_21= pd.read_csv(file_location + 'fifa_clean.csv', low_memory=False)


# In[26]:


# Let us view the dataset
fifa_21.head()


# In[27]:


# Let us view the dataset
fifa_21.tail()


# In[28]:


# Convert the specified columns to categorical datatype

fifa_21['W/F'] = fifa_21['W/F'].astype('category')
fifa_21['SM'] = fifa_21['SM'].astype('category')
fifa_21['A/W'] = fifa_21['A/W'].astype('category')
fifa_21['D/W'] = fifa_21['D/W'].astype('category')
fifa_21['IR'] = fifa_21['IR'].astype('category')

# Print the datatypes of the columns to confirm the conversion

print(fifa_21[['W/F', 'SM', 'A/W', 'D/W', 'IR']].dtypes)


# In[29]:


# Drop the column Positions since we have the Best Position column

# Drop the Positions column
fifa_21.drop('Positions', axis=1, inplace=True)


# In[30]:


# Remove special characters from Name, LongName, and Club columns

fifa_21['Name'] = fifa_21['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
fifa_21['LongName'] = fifa_21['LongName'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
fifa_21['Club'] = fifa_21['Club'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')


# In[31]:


# Save the fifa_21 clean dataframe as a CSV file to the specified path
fifa_21.to_csv('C:\\Users\\ragir\\OneDrive\\Desktop\\DATA ANALYTICS\\Portfolio Projects\\Data Cleaning Challenge\\fifa_21_clean.csv', index=False)

# Print a message to confirm the file has been saved
print('fifa_21.csv saved successfully.')


# In[ ]:





# In[ ]:




