#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning Challenge

# #### Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime 
import matplotlib.pyplot as plt


# #### Import dataset to Jupyter Notebook

# In[2]:


file_location = 'C:\\Users\\ragir\\OneDrive\\Desktop\\DATA ANALYTICS\\Portfolio Projects\\Data Cleaning Challenge\\'

fifa_data = pd.read_csv(file_location + 'fifa21 raw data v2.csv', low_memory=False)


# In[3]:


# Let us view the dataset
fifa_data.head()


# In[4]:


fifa_data.tail()


# In[5]:


# View number of rows and coumns in the dataset
# 18979 rows and 77 columns

fifa_data.shape


# In[6]:


# Check and delete duplicates in the dataset

# Check for duplicates
print(fifa_data.duplicated().sum())

# Drop duplicates
fifa_data.drop_duplicates(inplace=True)

# Check for duplicates after dropping
print(fifa_data.duplicated().sum())


# There are no duplicates in the dataset

# In[7]:


# Let's view all the column titles
fifa_data.columns


# In[8]:


# Rename the ↓OVA column to OVA

fifa_data = fifa_data.rename(columns={'↓OVA': 'OVA'})


# In[9]:


# Let's clean the ID column

# Check for datatype
print(fifa_data['ID'].dtype)

# Check for any missing values
print(fifa_data['ID'].isnull().sum())


# The ID column has int values and no mssing values

# In[10]:


# Let's clean the Name column

# Check for datatype
print(fifa_data['Name'].dtype)

# Check for any missing values
print(fifa_data['Name'].isnull().sum())

# Check for numerical data in the Name column
numerical_data = fifa_data['Name'].str.contains('\d+')

# Print rows with numerical data in the Name column
print(fifa_data[numerical_data])


# The Name column is of object datatype and has 0 missing values and no number values in the column

# In[11]:


# Let's clean the LongName column

# Check for datatype
print(fifa_data['LongName'].dtype)

# Check for any missing values
print(fifa_data['LongName'].isnull().sum())

# Check for numerical data in the Name column
numerical_data = fifa_data['LongName'].str.contains('\d+')

# Print rows with numerical data in the Name column
print(fifa_data[numerical_data])


# In[12]:


# Let's clean the photoUrl Column

# Check for datatype
print(fifa_data['photoUrl'].dtype)

# Check for any missing values
print(fifa_data['photoUrl'].isnull().sum())

# Check photo extension of each photo
fifa_data.photoUrl.head(10)


# In[13]:


# Check if all the photos are .png files
is_png = fifa_data['photoUrl'].str.endswith('.png').all()

# Print result
print(is_png)


# We use the str.endswith() method to check if all photo URLs in the photoUrl column end with the .png extension. We pass the string .png as the suffix to search for. The str.endswith() method returns a boolean series indicating whether each value in the photoUrl column ends with the suffix or not. We use the all() function to check if all the values are True.
# 
# All the photos in the photoUrl column are .png files

# In[14]:


# Let's clean the playerUrl column

# Check for datatype
print(fifa_data['playerUrl'].dtype)

# Check for any missing values
print(fifa_data['playerUrl'].isnull().sum())

# Check playerUrl data
fifa_data.playerUrl.head(10)


# There is no missing value in the playerUrl column

# In[15]:


# Let's clean the Nationality column

# Check for datatype
print(fifa_data['Nationality'].dtype)

# Check for any missing values
print(fifa_data['Nationality'].isnull().sum())

# Count unique values in the Nationality column
num_unique = fifa_data['Nationality'].nunique()

# Print result
print(num_unique)

# Check Nationality data
fifa_data.Nationality.unique()


# There are 164 different Nationalities in the Natioanlity column of the dataset

# In[16]:


# Create a new dataframe to add the cleaned columns above: ID, Name, LongName, photoURl, playerUrl, Nationality

# Clean columns
clean_data = fifa_data[['ID', 'Name', 'LongName', 'photoUrl', 'playerUrl', 'Nationality']]

# Create new dataframe
fifa_clean = pd.DataFrame(clean_data)

# Print new dataframe
print(fifa_clean)


# In[17]:


# Let us separate the clean columns from the dirty ones by creating a new dataset

other_data = fifa_data.drop(['ID', 'Name', 'LongName', 'photoUrl', 'playerUrl', 'Nationality'], axis=1)
fifa_dirty = pd.DataFrame(other_data)
print(fifa_dirty)


# In[18]:


# Let us separate and group the remaining columns into their datatypes for easy cleaning

# Create empty lists for each datatype
int_cols = []
str_cols = []
other_cols = []

# Loop through columns and categorize by datatype
for col in fifa_dirty.columns:
    if fifa_dirty[col].dtype == 'int64':
        int_cols.append(col)
    elif fifa_dirty[col].dtype == 'object':
        str_cols.append(col)
    else:
        other_cols.append(col)


# In[19]:


# Lets look at the integer columns

print(int_cols)

print("Count: ", len(int_cols))

# 53 columns are integers. These are easier to clean since they have less issues


# In[20]:


# Let's look at the int_cols as a group


fifa_dirty[int_cols].isna().any()
fifa_dirty[int_cols].isnull().any()

# we don't have any missing value
# lets look at the data types

fifa_dirty[int_cols].dtypes


# In[21]:


# Describing our data is the best way to check out for any issues in the int_cols

fifa_dirty[int_cols].describe()


# From the result we have no negative numbers, hence there are no issues with the int_cols

# In[22]:


# Let's add the int_cols into fifa_clean dataframe

# Create a list of integer columns in fifa_dirty
int_cols = [col for col in fifa_dirty.columns if fifa_dirty[col].dtype == 'int64']

# Select the integer columns from fifa_dirty
fifa_dirty_int = fifa_dirty[int_cols]

# Concatenate the integer columns with fifa_clean
fifa_clean = pd.concat([fifa_clean, fifa_dirty_int], axis=1)

# Verify that the integer columns have been added to fifa_clean
print(fifa_clean.info())


# In[23]:


# Let's look at the stri_cols and other_cols

print(str_cols)

print("Count: ", len(str_cols))

print(other_cols)

print("Count: ", len(other_cols))

# We have 18 str_cols and 0 other_cols in the fifa_dirty dataframe 


# In[24]:


# Let's clean the Club column

fifa_dirty['Club']


# In[25]:


# Clean the Club column by removing the whitespaces and other characters

fifa_dirty['Club'] = fifa_dirty['Club'].str.strip()

fifa_dirty['Club']


# In[26]:


# Add the Club column to the fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Club'])


# In[27]:


# Verify that the Club column has been added to fifa_clean

print(fifa_clean.info())


# In[28]:


# Let's clean the Positions column

fifa_dirty['Positions'] = fifa_dirty['Positions'].str.replace(', ', '/')


# In[29]:


# Add the Positions column to the fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Positions'])

# Verify that the Positions column has been added to fifa_clean

print(fifa_clean.info())


# In[30]:


print(fifa_clean['Positions'])


# In[31]:


# Let's clean the Height column

# Count unique values in the Height column
num_unique = fifa_dirty['Height'].nunique()

# Print result
print(num_unique)

# Check Height data
fifa_dirty.Height.unique()


# We see that some data is in centimeters while there's other data in feet.

# In[32]:


# Let's replace all data in feet into centimeters- replace ' with . and " with ''

fifa_dirty['Height'][fifa_dirty.Height.str.contains("'")]

fifa_dirty['Height'] = fifa_dirty.Height.str.replace("'", '.')
fifa_dirty['Height'] = fifa_dirty.Height.str.replace('"', '')

fifa_dirty.Height.unique()


# In[33]:


# Let's convert the data in feet into centimeters for consistency

def convert_height(height):
    if 'cm' in height:
        # height is already in centimeters
        return int(height.replace('cm', ''))
    else:
        # split the height in feet and inches using a period
        feet, inches = height.split(".")
        # convert feet and inches to centimeters
        height_cm = int(feet) * 30.48 + int(inches) * 2.54
        return height_cm

fifa_dirty['Height'] = fifa_dirty['Height'].apply(convert_height)


# In[34]:


fifa_dirty.Height.unique()


# In[35]:


# Add cm suffix to the Height column and in zero decimal places

fifa_dirty['Height'] = fifa_dirty['Height'].round(0).apply(lambda x: f"{int(x)}cm")


# In[36]:


fifa_dirty.Height.unique()


# In[37]:


# Add Height column to fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Height'])

# Verify that the Height column has been added to fifa_clean

print(fifa_clean.info())


# In[38]:


# Let's clean the Weight column

# Count unique values in the Height column
num_unique = fifa_dirty['Weight'].nunique()

# Print result
print(num_unique)

# Check Height data
fifa_dirty.Weight.unique()


# In[39]:


# Some data is in kgs and other in lbs.  Let's convert the data in lbs into kgs for consistency
# Loop through the "Weight" column and convert the values in lbs to kgs
for i in range(len(fifa_dirty)):
    if fifa_dirty.loc[i, 'Weight'][-3:] == 'lbs':
        weight_lb = float(fifa_dirty.loc[i, 'Weight'][:-3])
        weight_kg = weight_lb * 0.45359237
        fifa_dirty.loc[i, 'Weight'] = str(round(weight_kg)) + 'kg'


# Check Weight data
fifa_dirty.Weight.unique()


# In[40]:


# Add Weight column to fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Weight'])

# Verify that the Weight column has been added to fifa_clean

print(fifa_clean.info())


# In[41]:


# Let's check for nulls and NA values in the Height and Weight column in fifa_clean dataframe

# Check for nulls or NA values in the "Height" column

if fifa_clean['Height'].isnull().sum() > 0 or fifa_clean['Height'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Height' column.")
else:
    print("There are no nulls or NA values in the 'Height' column.")

# Check for nulls or NA values in the "Weight" column
if fifa_clean['Weight'].isnull().sum() > 0 or fifa_clean['Weight'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Weight' column.")
else:
    print("There are no nulls or NA values in the 'Weight' column.")

    
# This code uses the isnull() and isna() functions to check for nulls and NA values in the "Height" and "Weight" columns of the "fifa_clean" dataframe. If there are any nulls or NA values, it will print a message indicating that there are nulls or NA values in the respective column. Otherwise, it will print a message indicating that there are no nulls or NA values in the respective column.


# In[42]:


# Let's clean the Preferred Foot column

# Check for nulls or NA values in the "Preferred Foot" column
if fifa_dirty['Preferred Foot'].isnull().sum() > 0 or fifa_dirty['Preferred Foot'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Preferred Foot' column.")
else:
    print("There are no nulls or NA values in the 'Preferred Foot' column.")
    

# Count the number of unique values in the "Preferred Foot" column
num_unique = fifa_dirty['Preferred Foot'].nunique()

# Print the number of unique values
print("Number of unique values in 'Preferred Foot' column:", num_unique)


# Only left and right are the unique values in the Preferred Foot column


# In[43]:


# Add Preferred Foot column to fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Preferred Foot'])

# Verify that the Preferred Foot column has been added to fifa_clean

print(fifa_clean.info())


# In[44]:


# Let's clean the Best Position column

# Check for nulls and NAs in the Best Position column
if fifa_dirty['Best Position'].isnull().sum() > 0 or fifa_dirty['Best Position'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Best Position' column.")
else:
    print("There are no nulls or NA values in the 'Best Position' column.")
    

# Count the number of unique values in the "Preferred Foot" column
num_unique = fifa_dirty['Best Position'].nunique()

# Print the number of unique values
print("Number of unique values in 'Best Position' column:", num_unique)


# In[45]:


# Add Best Position column to fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Best Position'])

# Verify that the Preferred Foot column has been added to fifa_clean

print(fifa_clean.info())


# In[46]:


# Let's clean the Joined column

# Convert the "Joined" column to datetime datatype
fifa_dirty['Joined'] = pd.to_datetime(fifa_dirty['Joined'])

# Verify the datatype of the "Joined" column has been changed
print(fifa_dirty['Joined'].dtype)


# Check for nulls and NAs in the Joined column
if fifa_dirty['Joined'].isnull().sum() > 0 or fifa_dirty['Joined'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Joined' column.")
else:
    print("There are no nulls or NA values in the 'Joined' column.")
    


# In[47]:


# Add Joined column to fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Joined'])

# Verify that the Joined column has been added to fifa_clean

print(fifa_clean.info())


# In[48]:


# Let's clean the Loan Date End column

print(fifa_dirty['Loan Date End'])

# We observe that the column has a lot NaN values

# Convert the column datatype to datetime datatype
fifa_dirty['Loan Date End'] = pd.to_datetime(fifa_dirty['Loan Date End'])

print(fifa_dirty['Loan Date End'].dtype)


# In[49]:


# Create a new column to indicate players on loan and not on loan. This will justify the NaN values in the Loan Date End column

# Replace the NaN values with 0
fifa_dirty['Loan Date End'].fillna(0, inplace=True)


# In[50]:


# Create a new column that indicates if player is on loan or not

fifa_dirty['Loan'] = fifa_dirty['Loan Date End'].apply(lambda x: 'Not on Loan' if x == 0 else 'On Loan')


# In[51]:


print(fifa_dirty['Loan'])


# In[52]:


# Let's add Loan Date End column to fifa_clean

fifa_clean = fifa_clean.join(fifa_dirty['Loan Date End'])


print(fifa_clean.info())


# In[53]:


# Let's add Loan column to fifa_clean

fifa_clean = fifa_clean.join(fifa_dirty['Loan'])


print(fifa_clean.info())


# In[54]:


# Let's clean the Value column

print(fifa_dirty['Value'])

# Checkfor null and NA values in the column
if fifa_dirty['Value'].isnull().sum() > 0 or fifa_dirty['Value'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Value' column.")
else:
    print("There are no nulls or NA values in the 'Value' column.")
    
    
    
# Check currency in which the data is in - € 
currencies = fifa_dirty['Value'].str.extract('(\D+)').squeeze().unique()

print(currencies)


# In[55]:


# We observe that some data is in millions and other is in thousands

# Remove currency symbol
fifa_dirty['Value'] = fifa_dirty['Value'].str.replace('€', '')

# Replace K and M by multiplying
fifa_dirty['Value'] = fifa_dirty['Value'].apply(lambda x: float(x[:-1])*1e3 if x[-1]=='K' else (float(x[:-1])*1e6 if x[-1]=='M' else float(x)))

# Handle values with M or K after a decimal point
fifa_dirty['Value'] = fifa_dirty['Value'].apply(lambda x: x*1e6 if 'M' in str(x) and '.' not in str(x) else (x*1e3 if 'K' in str(x) and '.' not in str(x) else x))


# In[56]:


print(fifa_dirty['Value'])


# In[57]:


# Let's clean the Wage column - We will follow the same steps as in the Value column

print(fifa_dirty['Wage'])

# Checkfor null and NA values in the column
if fifa_dirty['Wage'].isnull().sum() > 0 or fifa_dirty['Wage'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Wage' column.")
else:
    print("There are no nulls or NA values in the 'Wage' column.")
    
    
    
# Check currency in which the data is in - € 
currencies = fifa_dirty['Wage'].str.extract('(\D+)').squeeze().unique()

print(currencies)


# In[58]:


# We observe that some data is in millions and other is in thousands

# Remove currency symbol
fifa_dirty['Wage'] = fifa_dirty['Wage'].str.replace('€', '')

# Replace K and M by multiplying
fifa_dirty['Wage'] = fifa_dirty['Wage'].apply(lambda x: float(x[:-1])*1e3 if x[-1]=='K' else (float(x[:-1])*1e6 if x[-1]=='M' else float(x)))

# Handle values with M or K after a decimal point
fifa_dirty['Wage'] = fifa_dirty['Wage'].apply(lambda x: x*1e6 if 'M' in str(x) and '.' not in str(x) else (x*1e3 if 'K' in str(x) and '.' not in str(x) else x))


# In[59]:


print(fifa_dirty['Wage'])


# In[60]:


# Let's clean the Release Clause column - We will follow the same steps as in the Value and Wage column

print(fifa_dirty['Release Clause'])

# Checkfor null and NA values in the column
if fifa_dirty['Release Clause'].isnull().sum() > 0 or fifa_dirty['Release Clause'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Release Clause' column.")
else:
    print("There are no nulls or NA values in the 'Release Clause' column.")
    
    
    
# Check currency in which the data is in - € 
currencies = fifa_dirty['Release Clause'].str.extract('(\D+)').squeeze().unique()

print(currencies)


# In[61]:


# We observe that some data is in millions and other is in thousands

# Remove currency symbol
fifa_dirty['Release Clause'] = fifa_dirty['Release Clause'].str.replace('€', '')

# Replace K and M by multiplying
fifa_dirty['Release Clause'] = fifa_dirty['Release Clause'].apply(lambda x: float(x[:-1])*1e3 if x[-1]=='K' else (float(x[:-1])*1e6 if x[-1]=='M' else float(x)))

# Handle values with M or K after a decimal point
fifa_dirty['Release Clause'] = fifa_dirty['Release Clause'].apply(lambda x: x*1e6 if 'M' in str(x) and '.' not in str(x) else (x*1e3 if 'K' in str(x) and '.' not in str(x) else x))


# In[62]:


print(fifa_dirty['Release Clause'])


# In[63]:


# Let's add the Value, Wage and Release Clause columns to the fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Value'])


fifa_clean = fifa_clean.join(fifa_dirty['Wage'])


fifa_clean = fifa_clean.join(fifa_dirty['Release Clause'])


# In[64]:


print(fifa_clean.info())


# In[65]:


# Let's clean the W/F Column

# Check for null and NA values in the column

if fifa_dirty['W/F'].isnull().sum() > 0 or fifa_dirty['W/F'].isna().sum() > 0:
    print("There are nulls or NA values in the 'W/F' column.")
else:
    print("There are no nulls or NA values in the 'W/F' column.")
    
# Check for unique values in the column  
fifa_dirty['W/F'].unique()
    
# We observe that there are no null or NaN values in the W/F column


    


# In[66]:


# Let us remove the stars in the column
# Since the column represents Weak Foot rating on a scale of 1-5, we can use the str.replace() function to replace the star with an empty string

fifa_dirty['W/F'] = fifa_dirty['W/F'].str.replace('★','')


# In[67]:


# Remove the space left after removing the star
fifa_dirty['W/F'] = fifa_dirty['W/F'].str.replace('★', '').str.strip()

fifa_dirty['W/F'].unique()


# In[68]:


# Let's clean the SM Column

# Check for null and NA values in the column

if fifa_dirty['SM'].isnull().sum() > 0 or fifa_dirty['SM'].isna().sum() > 0:
    print("There are nulls or NA values in the 'SM' column.")
else:
    print("There are no nulls or NA values in the 'SM' column.")
    
# Check for unique values in the column  
fifa_dirty['SM'].unique()
    
# We observe that there are no null or NaN values in the SM column


# In[69]:


# Let us remove the stars in the column
# Since the column represents SM rating on a scale of 1-5, we can use the str.replace() function to replace the star with an empty string

fifa_dirty['SM'] = fifa_dirty['SM'].str.replace('★','')

fifa_dirty['W/F'] = fifa_dirty['W/F'].str.replace('★', '').str.strip()

fifa_dirty['W/F'].unique()


# In[70]:


# Let's clean the SM Column

# Check for null and NA values in the column

if fifa_dirty['A/W'].isnull().sum() > 0 or fifa_dirty['A/W'].isna().sum() > 0:
    print("There are nulls or NA values in the 'A/W' column.")
else:
    print("There are no nulls or NA values in the 'A/W' column.")
    
# Check for unique values in the column  
fifa_dirty['A/W'].unique()
    
# We observe that there are no null or NaN values in the A/W column


# In[71]:


# Let's clean the D/W Column

# Check for null and NA values in the column

if fifa_dirty['D/W'].isnull().sum() > 0 or fifa_dirty['D/W'].isna().sum() > 0:
    print("There are nulls or NA values in the 'D/W' column.")
else:
    print("There are no nulls or NA values in the 'D/W' column.")
    
# Check for unique values in the column  
fifa_dirty['D/W'].unique()
    
# We observe that there are no null or NaN values in the A/W column


# In[72]:


# Let's clean the IR Column

# Check for null and NA values in the column

if fifa_dirty['IR'].isnull().sum() > 0 or fifa_dirty['IR'].isna().sum() > 0:
    print("There are nulls or NA values in the 'IR' column.")
else:
    print("There are no nulls or NA values in the 'IR' column.")
    
# Check for unique values in the column  
fifa_dirty['IR'].unique()
    
# We observe that there are no null or NaN values in the A/W column


# In[73]:


# Let us remove the stars in the column
# Since the column represents IR rating on a scale of 1-5, we can use the str.replace() function to replace the star with an empty string

fifa_dirty['IR'] = fifa_dirty['IR'].str.replace('★','')

fifa_dirty['IR'] = fifa_dirty['IR'].str.replace('★', '').str.strip()

fifa_dirty['IR'].unique()


# In[74]:


# Let's add the W/F, SM, A/W, D/W, IR columns to the fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['W/F'])


fifa_clean = fifa_clean.join(fifa_dirty['SM'])


fifa_clean = fifa_clean.join(fifa_dirty['A/W'])

fifa_clean = fifa_clean.join(fifa_dirty['D/W'])

fifa_clean = fifa_clean.join(fifa_dirty['IR'])


# In[75]:


print(fifa_clean.info())


# In[77]:


# Let's clean the Hits Column

# Check for null and NA values in the column

if fifa_dirty['Hits'].isnull().sum() > 0 or fifa_dirty['Hits'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Hits' column.")
else:
    print("There are no nulls or NA values in the 'Hits' column.")
    
# Check for unique values in the column  
fifa_dirty['Hits'].unique()
    
# We observe that there are nulls or NaN values in the Hits column and the data here is a bit messy
# We also notice that some data in the column is in thousands in the form of K

# Let's convert the K in thousands to 000s
# replace the 'K' suffix with '000' in the "Hits" column
fifa_dirty["Hits"] = fifa_dirty["Hits"].str.replace('K', '000')

# convert the "Hits" column to numeric type
fifa_dirty["Hits"] = pd.to_numeric(fifa_dirty["Hits"], errors='coerce')

# print the first 10 rows of the modified "Hits" column
print(fifa_dirty["Hits"].head(10))


# In[78]:


# count the number of null and NaN values in the "Hits" column
num_null = fifa_dirty["Hits"].isnull().sum()
num_nan = np.isnan(fifa_dirty["Hits"]).sum()

# print the results
print("Number of null values:", num_null)
print("Number of NaN values:", num_nan)


# In[79]:


# We can replace the Null values with 0 in the Hits column

fifa_dirty["Hits"] = fifa_dirty["Hits"].fillna(0)


# print the results
print(fifa_dirty['Hits'])


# In[80]:


# Let's add the Hits column to the fifa_clean dataframe

fifa_clean = fifa_clean.join(fifa_dirty['Hits'])


print(fifa_clean.info())


# In[81]:


# Let's clean the Contract column

# Check for null and NA values in the column

if fifa_dirty['Contract'].isnull().sum() > 0 or fifa_dirty['Contract'].isna().sum() > 0:
    print("There are nulls or NA values in the 'Contract' column.")
else:
    print("There are no nulls or NA values in the 'Contract' column.")
    
# Check for unique values in the column  
fifa_dirty['Contract'].unique()


# In[82]:


#explore the players with free contract

fifa_dirty[fifa_dirty.Contract=='Free']


# In[84]:


# Explore players that are on loan

on_loan=fifa_dirty[fifa_dirty.Contract.str.contains("On Loan")]
on_loan


# In[94]:


# From the exploration, it shows their contract is the same as loan end date which is kind of irrelevant keeping both

# Check the number of players that are on loan
print(len(on_loan))

# Count the number of non-zero values in 'Loan Date End' column
num_loans = (fifa_dirty['Loan Date End'] != 0).astype(bool).sum()

# Print the result
print(f"There are {num_loans} non-zero values in the 'Loan Date End' column.")


# In[95]:


# Define the contract_sorting() function

def contract_sorting(Contract):
    if Contract == 'Free':
        contract_type = 'Free Agent'
        contract_start = None
        contract_end = None
    elif ' On Loan' in Contract:
        contract_type = 'On Loan'
        contract_start= None
        contract_end= Contract.replace(' On Loan','')
    else:
        contract_type = 'On Contract'
        contract_start, contract_end = Contract.split(' ~ ')

    return contract_start, contract_end,contract_type

# Apply the function to the 'Contract' column of the fifa_dirty dataframe
fifa_dirty[['Contract Start', 'Contract End', 'Contract Type']] = fifa_dirty['Contract'].apply(lambda x: pd.Series(contract_sorting(x)))

# Print the first few rows of the modified dataframe to check the result
print(fifa_dirty.head())


# In[99]:


print(fifa_dirty['Contract Start'])

print(fifa_dirty['Contract End'])

print(fifa_dirty['Contract Type'])


# In[101]:


# Using the apply method on contract_sorting and apply method to put it into three series and assigning the values to newly created columns

fifa_dirty[['Contract Start','Contract End', 'Contract Type']]=fifa_dirty.Contract.apply(contract_sorting).apply(pd.Series)


# In[103]:


fifa_clean = fifa_clean.join(fifa_dirty['Contract Start'])

fifa_clean = fifa_clean.join(fifa_dirty['Contract End'])

fifa_clean = fifa_clean.join(fifa_dirty['Contract Type'])


print(fifa_clean.info())


# In[106]:


# Dropping irrelevant columns from fifa_clean

fifa_clean.drop(['Loan Date End', 'Loan'],axis=1,inplace=True)


# In[108]:


print(fifa_clean.info())


# We have managed to clean all the relevant columns in the dataset and get it ready for exploration, analysis and visualization.

# In[121]:


# Rename some columns 
fifa_clean.rename(columns={'Height':'Height (cm)',
                  'Weight':'Weight (kg)',
                  'Value':'Value (Euro)',
                  'Wage':'Wage (Euro)',
                  'Release Clause':'Release Clause (Euro)'}, inplace=True)


# In[122]:


# Save the fifa_clean dataframe as a CSV file to the specified path
fifa_clean.to_csv('C:\\Users\\ragir\\OneDrive\\Desktop\\DATA ANALYTICS\\Portfolio Projects\\Data Cleaning Challenge\\fifa_clean.csv', index=False)

# Print a message to confirm the file has been saved
print('fifa_clean.csv saved successfully.')


# We have exported the clean data as a csv file which we can now analyze in any other tool easily and visualize the data in BI tools as well.

# In[ ]:





# In[ ]:




