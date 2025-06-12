import pandas as pd
import numpy as np

dataset = pd.read_csv('Dataset .csv')

## To check the dataset when changes are made to dataset
def printData(number):
    print(dataset.head(number))

# printData(5)

## Dropping the unnecessary features from the table
def dropFeature(data,feature_name):
    data.drop([feature_name], axis = 1, inplace = True)

dropFeature(dataset,'Restaurant ID')
dropFeature(dataset,'Restaurant Name')
dropFeature(dataset,'Country Code')
dropFeature(dataset,'City')
dropFeature(dataset,'Address')
dropFeature(dataset,'Locality')
dropFeature(dataset,'Locality Verbose')
dropFeature(dataset,'Longitude')
dropFeature(dataset,'Latitude')

# printData(5)

## Just to check if the dataset has any empty set
def emptySet():
    if dataset.isnull().values.any():
        print('Yes')
    else:
        print('No')

# emptySet()

## To check which features have empty set
def emptyfeature(data, feature_name):
    if data[feature_name].isnull().values.any():
        print(dataset[feature_name].isnull().sum())
    else:
        print('No empty values')

# emptyfeature(dataset, 'Cuisines') # 9
# emptyfeature(dataset, 'Average Cost for two') # 0
# emptyfeature(dataset, 'Currency') # 0
# emptyfeature(dataset, 'Has Table booking') # 0
# emptyfeature(dataset, 'Has Online delivery') # 0 
# emptyfeature(dataset, 'Is delivering now') # 0 
# emptyfeature(dataset, 'Switch to order menu') # 0
# emptyfeature(dataset, 'Price range') # 0
# emptyfeature(dataset, 'Aggregate rating') # 0
# emptyfeature(dataset, 'Rating color') # 0
# emptyfeature(dataset, 'Rating text') # 0
# emptyfeature(dataset, 'Votes') # 0

## Filling the 'Cuisines' feature's empty values as "Unknown"
dataset['Cuisines'] = dataset['Cuisines'].fillna('Unknown')

## Count Unique values in a feature
def countUniqueValues(data,feature_name):
    print(data[feature_name].nunique())

# countUniqueValues(dataset,'Cuisines') # 1825
# countUniqueValues(dataset,'Currency') # 12 - Onehotencoding
# countUniqueValues(dataset,'Rating color') # 6 - Onehotencoding
# countUniqueValues(dataset,'Rating text') # 6 - Onehotencoding

# print(dataset.columns)

## Mapping the binary categorical values into numeric values
def mappingFeatures(data,feature_name):
    data[feature_name] = data[feature_name].map({'Yes': 1, 'No': 0})

mappingFeatures(dataset,'Has Table booking')
mappingFeatures(dataset,'Has Online delivery')
mappingFeatures(dataset,'Is delivering now')
mappingFeatures(dataset,'Switch to order menu')

# printData(5)

## 'Cuisines' has too many categorical values for onehotencoding, have to convert it into a new cuisine list with only important information
dataset['Cuisines'] = dataset['Cuisines'].str.split(',').str[0].str.strip()

# countUniqueValues(dataset,'Cuisines') # 119 - onehotencoding

X = dataset.iloc[:, [i for i in range(dataset.shape[1]) if i != 8]]
y = dataset.iloc[:, 8]

feature_names = X.columns.tolist()

## One hot encoding the categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encode_features = ['Cuisines', 'Currency', 'Rating color', 'Rating text']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), encode_features)], remainder='passthrough')
X = np.array(ct.fit_transform(X))

encoded_col_names = ct.named_transformers_['encoder'].get_feature_names_out(encode_features)

non_encoded = [col for col in dataset.columns if col not in encode_features + ['Aggregate rating']]
all_feature_names = list(encoded_col_names) + non_encoded

X_df = pd.DataFrame(X, columns=all_feature_names)

## Splitting Data for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_df,y,test_size=0.2,random_state=0)

## Preparing the model for fitting and prediction
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

## Evaluation metrics
from sklearn.metrics import r2_score, root_mean_squared_error
print('RMSE:', root_mean_squared_error(y_test,y_pred))
print('R2:', r2_score(y_test,y_pred))

## Analyzing the most influential feature
coefficients = pd.Series(model.coef_, index=X_train.columns)
print("Top Influential Features:")
print(coefficients.sort_values(ascending=False).head(10))

print("\nLeast Influential Features:")
print(coefficients.sort_values().head(10))