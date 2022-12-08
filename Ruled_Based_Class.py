import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 900)

df = pd.read_csv('DirectMarketing.csv')
df.head()


###  DATA CLEANING AND FEATURE ENGINEERING ###

# Removing unnecessary columns
df.drop(['History', 'Catalogs'], axis=1, inplace=True)

# There is any null values on dataset
df.isnull().sum()

# Age Column
df['Age'].value_counts()
sns.countplot(df['Age'])
plt.show()

# Create FamSize column
print("Total categories in the feature Marital_Status:\n",
      df["Married"].value_counts(), "\n")

df['Married'].replace({'Married': 2, 'Single': 1}, inplace=True)

df['FamSize'] = df['Married'] + df['Children']
df.drop(['Married', 'Children'], axis=1, inplace=True)

# Turning Salary Column Categorical

df['Salary_Lev'] = pd.qcut(df['Salary'], 5, labels=['E', 'D', 'C', 'B', 'A'])
df.groupby('Salary_Lev').agg({'AmountSpent': 'sum'})


### DATA ANALYSIS and VISUALIZATION ###

sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, target, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        "TARGET_COUNT": dataframe.groupby(col_name)[target].count(),
                        "TARGET_MEAN": dataframe.groupby(col_name)[target].mean()}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, 'AmountSpent', plot=True)

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

age_gen = df.groupby(['Age', 'Gender']).agg({'AmountSpent': ['mean']})
age_gen = age_gen.reset_index()
age_gen.head()
sns.barplot(x=age_gen['Age'], y=age_gen['AmountSpent']['mean'], hue=age_gen['Gender'], data=df)
plt.show()

own_loc = df.groupby(['OwnHome', 'Location']).agg({'AmountSpent': ['mean']})
own_loc = own_loc.reset_index()
own_loc.head()
sns.barplot(x=own_loc['OwnHome'], y=own_loc['AmountSpent']['mean'], hue=own_loc['Location'], data=df)
plt.show()

fam_size = df.groupby(['FamSize']).agg({'AmountSpent': ['mean']})
fam_size = fam_size.reset_index()
fam_size.head()
sns.barplot(x=fam_size['FamSize'], y=fam_size['AmountSpent']['mean'], data=df)
plt.show()


##  Ruled_Based_Classification

agg_df = df.groupby(['Age', 'Gender', 'OwnHome', 'Location', 'FamSize', 'Salary']).agg({'AmountSpent': 'mean'})\
    .sort_values(by='AmountSpent', ascending=False)

agg_df = agg_df.reset_index()

agg_df['CUSTOMER_LEVEL_BASED'] = [row[0].upper() + '_' + row[1].upper() + '_' +
                                  row[2].upper() + '_' + row[3].upper() + '_' + str(row[4])
                              for row in agg_df.values]

agg_df = agg_df.groupby('CUSTOMER_LEVEL_BASED').agg({'AmountSpent': 'mean'})

agg_df = agg_df.reset_index()

agg_df['CUSTOMER_LEVEL_BASED'].value_counts()


new_user = 'OLD_MALE_OWN_FAR_2'
agg_df[agg_df['CUSTOMER_LEVEL_BASED'] == new_user]

agg_df.head()