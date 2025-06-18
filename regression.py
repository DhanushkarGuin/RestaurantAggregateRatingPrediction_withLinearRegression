import pandas as pd

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

# print(dataset.isnull().sum())  # Cuisines = 9

## Reducing Cuisines' unique values
dataset['Cuisines'] = dataset['Cuisines'].str.split(',').str[0].str.strip()

X = dataset.drop(columns=['Aggregate rating'])
y = dataset['Aggregate rating']

## Splitting data for Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

## Pipelining
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

categorical_features = ['Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery',
                        'Is delivering now', 'Rating color', 'Rating text']
numerical_features = ['Average Cost for two', 'Price range', 'Votes']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

## Metrics Evaluation
from sklearn.metrics import root_mean_squared_error,r2_score
print('RMSE', root_mean_squared_error(y_test,y_pred))
print('R2', r2_score(y_test,y_pred))

import pickle
pickle.dump(pipeline,open('pipeline.pkl', 'wb'))

print(dataset.columns.tolist())