import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor


##This go through will solely be based on the XGBoost model. This is merely a fraction of the possibilities with this dataset. Having said that if a user would go through this
#dataset themselves, I would recommend using statistical approaches to find feature importance and then only keep some of the important features. 
#Another thing that could be done, is testing different model and then having the end value as some sort of median between the results.

test_df = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/House pricing/test.csv', index_col='Id')
#Training data
df = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/House pricing/train.csv', index_col='Id')

#Checking the data
#print(df.info())
#print(df.head())

#####exploratory data analysis######

#In the begining we drop features with too many missing values
df.loc[:, df.isna().mean() > 0.4].isna().mean()

df = df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

#As we use XGBoost we know that we need to define the target variable - in this case it is salesprice

X = df.drop('SalePrice', axis=1)
y = np.log(df['SalePrice'])

#Defining numerical and categorical features:
#Categorical values with no order
nominal_features = [
    'MSSubClass', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'SaleType', 
    'SaleCondition','GarageType'
]
#Categorical values with some sort of order
ordinal_features = [
    'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 
    'ExterCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 
    'Electrical', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
    'GarageFinish', 'GarageQual', 'GarageCond'
]
#Continuous numerical values.
continuous_features = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
    'MiscVal'
]
#Discrete numerical values.
discrete_features = [
    'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
    'MoSold', 'YrSold'
]

#Filling missing values for categorical values with None
for col in (nominal_features + ordinal_features):
    X[col] = X[col].fillna('None')
#Filling missing values for numerical values with 0
for col in (continuous_features + discrete_features):
    X[col] = X[col].fillna(0)

#One-hot encoding - as most ML models (except catboost) can't handle categorical values, we perform the so called one-hot encoding which assigns values 1 and 0 for categorical variables.

dummies = pd.get_dummies(X[nominal_features]).sort_index()

X = pd.concat([X, dummies], axis=1)
X = X.drop(nominal_features, axis=1)

print(X.info())
print(X.head())
#Performing ordinal-encoding:
rating = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

ordinal_encoding = {
    'LotShape': {'None': 0, 'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}, 
    'Utilities': {'None': 0, 'ElO': 1, 'NoSeWa': 2, 'NoSeWr': 3, 'AllPub': 4}, 
    'LandSlope': {'None': 0, 'Gtl': 1, 'Mod': 2, 'Sev': 3}, 
    'ExterQual': rating, 
    'ExterCond': rating, 
    'BsmtQual': rating, 
    'BsmtCond': rating, 
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 
    'HeatingQC': rating, 
    'CentralAir': {'None': 0, 'N': 1, 'Y': 2}, 
    'Electrical': {'None': 0, 'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 
    'KitchenQual': rating, 
    'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 
    'GarageQual': rating, 
    'GarageCond': rating, 
    'PavedDrive': {'None': 0, 'N': 1, 'P': 2, 'Y': 3}
}
X = X.replace(ordinal_encoding)
#Check the data
#print(X.info())
#print(X.head())

#Scaling features such that all have same scale:

X[:] = RobustScaler().fit_transform(X)

print(X.info())
print(X.head())

#prepare features function for training (basically all above)
def prepare_features(df: pd.DataFrame, feature_names: list = None):
    """Preparing features for training"""
    
    # Creating DF
    
    if feature_names is not None:
        X = pd.concat([pd.DataFrame(columns=feature_names), df])
    else:
        X = df
    
    # Defining numerical and categorical features
    
    nominal_features = [
        'MSSubClass', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'SaleType', 
        'SaleCondition','GarageType'
    ]

    ordinal_features = [
        'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 
        'ExterCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 
        'Electrical', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
        'GarageFinish', 'GarageQual', 'GarageCond'
    ]

    continuous_features = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
        'MiscVal'
    ]

    discrete_features = [
        'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
        'MoSold', 'YrSold'
    ]
    
    # Filling missing categorical values with None

    for col in (nominal_features + ordinal_features):
        X[col] = X[col].fillna('None')

    # Filling numerical missing values with 0

    for col in (continuous_features + discrete_features):
        X[col] = X[col].fillna(0)
    
    # One Hot Encoding

    dummies = pd.get_dummies(X[nominal_features]).sort_index()
    dummies_cols = list(set(dummies.columns) & set(X.columns))
    X[dummies_cols] = dummies[dummies_cols]
    X = X.drop(nominal_features, axis=1)
    
    # Ordinal Encoding

    rating = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    ordinal_encoding = {
        'LotShape': {'None': 0, 'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}, 
        'Utilities': {'None': 0, 'ElO': 1, 'NoSeWa': 2, 'NoSeWr': 3, 'AllPub': 4}, 
        'LandSlope': {'None': 0, 'Gtl': 1, 'Mod': 2, 'Sev': 3}, 
        'ExterQual': rating, 
        'ExterCond': rating, 
        'BsmtQual': rating, 
        'BsmtCond': rating, 
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 
        'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 
        'HeatingQC': rating, 
        'CentralAir': {'None': 0, 'N': 1, 'Y': 2}, 
        'Electrical': {'None': 0, 'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 
        'KitchenQual': rating, 
        'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 
        'GarageQual': rating, 
        'GarageCond': rating, 
        'PavedDrive': {'None': 0, 'N': 1, 'P': 2, 'Y': 3}
    }

    X = X.replace(ordinal_encoding)
    
    # Feature selection
    
    if feature_names is not None:
        X = X[feature_names]
    
    # Filling NAs
    
    X = X.fillna(0)
    
    # Scaling features
    
    X[:] = RobustScaler().fit_transform(X)
    
    return X

#Defining plotting function for later purposes.
def plot_actual_vs_pred(model, X, y):
    """Plotting actual vs predicted label"""
    
    y_pred = np.exp(model.predict(X))
    
    plot_data = pd.concat([np.exp(y), pd.Series(y_pred, name='PredictedPrice', index=y.index)], axis=1)
    plot_data = plot_data.sort_values('SalePrice')
    plot_data.index = y.index
    plot_data = plot_data.reset_index()
    
    fig = px.scatter(plot_data, x='Id', y='SalePrice')
    fig.add_trace(go.Scatter(x=plot_data['Id'], y=plot_data['PredictedPrice'], name='Prediction'))
    fig.show()

##### Performing the XGBoost model #####
#train the model on already defined train data
model = XGBRegressor(n_estimators=800, learning_rate=0.03).fit(X, y)

print('\nModel score:', np.mean(cross_val_score(model, X, y)))
#Plot forecast vs predicted
#plot_actual_vs_pred(model, X, y)

#Preparing to perform XGBoost on our test data
print(test_df.head())
#Feature importance
X_test = prepare_features(test_df, X.columns)

print(X_test.head())

#Predicting house prices for training set
test_preds = pd.DataFrame.from_dict({'Id': test_df.index,'SalePrice': np.exp(model.predict(X_test))}).astype(np.int64)

print(test_preds.head())
#Merge the datasets for predicted house prices and original dataset. 
Forecasted_houseprices = pd.merge(test_df, test_preds, on ="Id")

print(Forecasted_houseprices.head())

Forecasted_houseprices.to_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/House pricing/Predicted_houseprices.csv', index=False)