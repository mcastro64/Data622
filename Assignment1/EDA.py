import pandas as pd
import numpy as np
import seaborn as sns                       
import matplotlib.pyplot as plt

banks_df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')

# first 5 rows
print(banks_df.head())


# number of rows and cols
print('\nShape of dataframe')
print(banks_df.shape)

# column types
print('\nColumn Types')
print(banks_df.dtypes)


# df summary
print('\nSummary metrics for numerical fields')
print(banks_df.describe())

# check null rows
print('\n# of Null Rows')
print(banks_df.isnull().sum())


# Select categorical columns
categorical_cols = banks_df.select_dtypes(include=['object']).columns
noncategorical_cols = banks_df.select_dtypes(exclude=['object']).columns

# generate bank profile
banks_df.info()

# Convert selected object columns to 'category'
# and print value counts
for i, col in enumerate(categorical_cols):
    banks_df[col] = banks_df[col].astype('category')
    val_counts = banks_df[col].value_counts()
    print(f"\nValues for column {val_counts}")

# drop column 'day_of_week' since it doesn't add anything (values fairly even);
# we will keep `month` for the moment in case it offers insights into seasonality
banks_df = banks_df.drop(columns=['day_of_week'])

# reassign values
categorical_cols = banks_df.select_dtypes(include=['category']).columns



# ProfileReport(banks_df)

# print facetterd barplots for categorical values
g = sns.catplot(
    data=banks_df.melt(value_vars=categorical_cols), 
    kind='count',
    x='value',
    hue='y',
    col='variable',
    col_wrap=4, 
    height=4,
    aspect=1,
    sharex=False
)

g.set_xticklabels(rotation=45, ha="right")

plt.show()



# show pair plots for 
sns.pairplot(banks_df)


fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(15, 8))

x_counter = 0
y_counter = 0
for i, col in enumerate(noncategorical_cols):
    sns.boxplot(data=banks_df, x=col, y='y', ax=axes[y_counter, x_counter])
    #axes[i].set_title(f'Countplot for {col}')
    #axes[i].tick_params(axis='x', rotation=45)
    x_counter += 1
    if x_counter == 4:
        x_counter = 0
        y_counter += 1
    if i == len(noncategorical_cols):
        break
    
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(15, 8))

x_counter = 0
y_counter = 0
for i, col in enumerate(noncategorical_cols):
    sns.histplot(data=banks_df, x=col, ax=axes[y_counter, x_counter])
    #axes[i].set_title(f'Countplot for {col}')
    #axes[i].tick_params(axis='x', rotation=45)
    x_counter += 1
    if x_counter == 4:
        x_counter = 0
        y_counter += 1
    
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
c= banks_df[noncategorical_cols].corr()
sns.heatmap(c, cmap="BrBG", annot=True)
print(c)

