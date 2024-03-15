import numpy as np
from sklearn.preprocessing import  OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

def dataset_split(X,Y,test_size :float):
    data_split = train_test_split(X, Y, random_state=42,test_size=test_size)
    return data_split

def one_hot_encode(X_train, X_test)-> tuple:
    enc = OneHotEncoder(handle_unknown='ignore')

    
    X_train = enc.fit_transform(X_train)

    # Transform the testing data using the same encoder
    X_test = enc.transform(X_test)

    return X_train, X_test

def one_hot_encode_dataframe(df):
    
    
    # Perform one-hot encoding
    df_enc = pd.get_dummies(df)

    return df_enc

def standard_scaling(X_train,X_test)->tuple:
    
    scaler=StandardScaler(with_mean=False)
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    return X_train , X_test

def remove_outliers_pos(df,column_name :str):
# IQR dla klasy 1
    Q1 = np.percentile(df[df['Attrition'] == 1][f'{column_name}'], 25, interpolation='midpoint')
    
    Q3 = np.percentile(df[df['Attrition'] == 1][f'{column_name}'], 75,
                    interpolation = 'midpoint')

    IQR = Q3 - Q1
    
    print("Old Shape: ", df.shape)
    
    # Górny przedział
    upper = np.where(df[df['Attrition'] == 1][f'{column_name}'] >= (Q3+1.5*IQR))
    
    # Dolny przedział
    lower = np.where(df[df['Attrition'] == 1][f'{column_name}'] <= (Q1-1.5*IQR))
    
    # Usuwamy wartości odstające
    df.drop(upper[0], inplace = True)
    df.drop(lower[0], inplace = True)
    
    print("New Shape: ", df.shape)
    
    sns.boxplot(x=f'{column_name}',hue="Attrition", data=df)

def remove_outliers_neg(df,column_name :str):
# IQR dla klasy 1
    Q1 = np.percentile(df[df['Attrition'] == 0][f'{column_name}'], 25, interpolation='midpoint')
    
    Q3 = np.percentile(df[df['Attrition'] == 0][f'{column_name}'], 75,
                    interpolation = 'midpoint')

    IQR = Q3 - Q1
    
    print("Old Shape: ", df.shape)
    
    # Górny przedział
    upper = np.where(df[df['Attrition'] == 0][f'{column_name}'] >= (Q3+1.5*IQR))
    
    # Dolny przedział
    lower = np.where(df[df['Attrition'] == 0][f'{column_name}'] <= (Q1-1.5*IQR))
    
    # Usuwamy wartości odstające
    df.drop(upper[0], inplace = True)
    df.drop(lower[0], inplace = True)
    
    print("New Shape: ", df.shape)
    
    sns.boxplot(x=f'{column_name}',hue="Attrition", data=df)

def  get_value_distribution(Y) -> tuple:
    # sprwadzamy zbilansowanie zbioru
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()

    print(f"Number of positive: {n_pos}, number of negative: {n_neg}")

    return n_pos,n_neg

def transorm_data(df):
    df_numeric=df._get_numeric_data()
    numeric_cols=list(df_numeric.columns)
    df_categorical=df.drop(numeric_cols,axis=1)
    
    #scaler
    scaler = StandardScaler()
    df_numeric = pd.DataFrame(scaler.fit_transform(df_numeric))
    #zamiana nazwy kolum
    df_numeric.columns=numeric_cols

    #one hot encoding
    df_categorical=df_categorical.astype("string")
    df_categorical=pd.get_dummies(df_categorical)
    #zamiana nazw kolumn na string
    df_categorical.columns=df_categorical.columns.astype(str)

    df_ready = pd.concat([df_numeric, df_categorical], axis=1, join='inner')
    
    return df_ready

def scale_df(df):
    df_numeric=df._get_numeric_data()
    numeric_cols=list(df_numeric.columns)
    #scaler
    scaler = StandardScaler()
    df_numeric = pd.DataFrame(scaler.fit_transform(df_numeric))
    #zamiana nazwy kolum
    df_numeric.columns=numeric_cols
