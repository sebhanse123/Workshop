from email.utils import parsedate_to_datetime
from turtle import title
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from plotly.offline import init_notebook_mode, iplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
#!pip install xgboost
import xgboost as xgb
import copy

test = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/Item and Demand/test.csv')
train = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/Item and Demand/train.csv')
sample = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/Item and Demand/sample_submission.csv')
#Copy
test1 = copy.deepcopy(test)
train1 = copy.deepcopy(train)
#Check the data
#print(test.head())
#print(train.head())


################# Data preprocessing ########################


#Feature engineering for ML models/adding extra features such that some ML models can cope with seasonality and dates. In general ML models are not too happy for the datetype: dates
#thus below we convert the original field "date" too many numerical values for dates. Fx. month will be in the range 1-12. Furthermore, the numerical added values will provide more
#info that can be added to the model. 

train['date']=pd.to_datetime(train['date'], infer_datetime_format=True)
test['date']=pd.to_datetime(test['date'], infer_datetime_format=True)
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
#train['week'] = train['date'].dt.week
#train['weekofyear'] = train['date'].dt.weekofyear
train['dayofweek'] = train['date'].dt.dayofweek
train['weekday'] = train['date'].dt.weekday
train['dayofyear'] = train['date'].dt.dayofyear
train['quarter'] = train['date'].dt.quarter
train['is_month_start'] = train['date'].dt.is_month_start
train['is_month_end'] =train['date'].dt.is_month_end
train['is_quarter_start'] = train['date'].dt.is_quarter_start
train['is_quarter_end'] = train['date'].dt.is_quarter_end
train['is_year_start'] = train['date'].dt.is_year_start
train['is_year_end'] = train['date'].dt.is_year_end
train['daily_avg']=train.groupby(['item','store','dayofweek'])['sales'].transform('mean')
train['monthly_avg']=train.groupby(['item','store','month'])['sales'].transform('mean')
train["mean_store_item_month"] = train.groupby(['month',"item","store"])["sales"].transform("mean")
train["item_month_sum"] = train.groupby(['month',"item"])["sales"].transform("sum") # total sales of that item  for all stores
train["store_month_sum"] = train.groupby(['month',"store"])["sales"].transform("sum") # total sales of that store  for all items
daily_avg=train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg=train.groupby(['item','store','month'])['sales'].mean().reset_index()
mean_store_item_month = train.groupby(['month','item','store'])['sales'].mean().reset_index()
item_month_sum=train.groupby(['month','item'])['sales'].sum().reset_index()
store_month_sum=train.groupby(['month','store'])['sales'].sum().reset_index()
#print(train.dtypes)

#Convert bool true and false to (1 and 0). Why do this? Well most ML models (except for a method called catboost) do not content with categorical variables (bools). 
train['is_month_start'] = train['is_month_start'].replace({True: 1, False: 0})
train['is_month_end'] = train['is_month_end'].replace({True: 1, False: 0})
train['is_quarter_start'] = train['is_quarter_start'].replace({True: 1, False: 0})
train['is_quarter_end'] = train['is_quarter_end'].replace({True: 1, False: 0})
train['is_year_start'] = train['is_year_start'].replace({True: 1, False: 0})
train['is_year_end'] = train['is_year_end'].replace({True: 1, False: 0})

#convert float to int
train['daily_avg']=train['daily_avg'].astype(np.int64)
train['monthly_avg']=train['monthly_avg'].astype(np.int64)
train['mean_store_item_month']=train['mean_store_item_month'].astype(np.int64)
#train['store_item_shifted_365']=train['store_item_shifted_365'].astype(np.int64)
#print(train.dtypes)
#print(train.head())

### test for missing values - Notice that this is a nice time series, try and imagine what would be required if there were missing values? An example for missing values could
# for example be if you were to try and model revenue accross a business unit on customer level. We do not have the same customer for perpetuity, or even generate a revenue
# for this customer each month.
train_NA = train.isna()
train_num_NA = train_NA.sum()
#print(train_num_NA)

## Repeat for test
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
#test['week'] = test['date'].dt.week
#test['weekofyear'] = test['date'].dt.weekofyear
test['dayofweek'] = test['date'].dt.dayofweek
test['weekday'] = test['date'].dt.weekday
test['dayofyear'] = test['date'].dt.dayofyear
test['quarter'] = test['date'].dt.quarter
test['is_month_start'] = test['date'].dt.is_month_start
test['is_month_end']= test['date'].dt.is_month_end
test['is_quarter_start'] = test['date'].dt.is_quarter_start
test['is_quarter_end'] = test['date'].dt.is_quarter_end
test['is_year_start'] = test['date'].dt.is_year_start
test['is_year_end'] = test['date'].dt.is_year_end
test.dtypes

#Creating mask for boolean values for test dataset:
test['is_month_start'] = test['is_month_start'].replace({True: 1, False: 0})
test['is_month_end'] = test['is_month_end'].replace({True: 1, False: 0})
test['is_quarter_start'] = test['is_quarter_start'].replace({True: 1, False: 0})
test['is_quarter_end'] = test['is_quarter_end'].replace({True: 1, False: 0})
test['is_year_start'] = test['is_year_start'].replace({True: 1, False: 0})
test['is_year_end'] = test['is_year_end'].replace({True: 1, False: 0})
test['is_year_end'] = test['is_year_end'].astype(str).astype(np.int64)
#print(test.head())

#Add sales column with merge/join function to test data
def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    
    x=x.rename(columns={'sales':col_name})
    return x

test=merge(test, daily_avg,['item','store','dayofweek'],'daily_avg')
test=merge(test, monthly_avg,['item','store','month'],'monthly_avg')
test=merge(test, mean_store_item_month, ['month','item','store'],'mean_store_item_month')
test=merge(test, item_month_sum,['month','item'],'item_month_sum')
test=merge(test, store_month_sum,['month','store'],'store_month_sum')
#test=merge(test, store_item_shifted_365,['item','store'],'store_item_shifted_365')
#test=merge(test, item_week_shifted_90,['weekofyear', 'item'],'item-week_shifted_90')

#convert float to int
test['daily_avg']=test['daily_avg'].astype(np.int64)
test['monthly_avg']=test['monthly_avg'].astype(np.int64)
test['mean_store_item_month']=test['mean_store_item_month'].astype(np.int64)
#test['store_item_shifted_365']=test['store_item_shifted_365'].astype(np.int64)
test.dtypes
#print(test.head())
#test for missing values:
train_NA = train.isna()
train_num_NA = train_NA.sum()
#print(train_num_NA)

## Check date range.
print('Min date from train set: %s' % train['date'].min().date())
print('Max date from train set: %s' % train['date'].max().date())

##Finding lag size
lag_size = (test['date'].max().date() - train['date'].max().date()).days
print('Max date from train set: %s' % train['date'].max().date())
print('Max date from test set: %s' % test['date'].max().date())
print('Forecast lag size', lag_size)

#### Plotting histogram
plt.hist(train['sales'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.title('Histogram of sales')
plt.xlabel('sales')
plt.ylabel('sales frequency')

print('Skew Dist:',train['sales'].skew())
print('Kurtosis Dist:',train['sales'].kurt())
#Data is skewed to the right. 

############### Data visualization ###############

#Total sales accros business
daily_sales = train.groupby('date', as_index=False)['sales'].sum()
daily_sales.set_index('date', inplace=True)
#Total sales accross stores
store_daily_sales = train.groupby(['store', 'date'], as_index=False)['sales'].sum()
#Total sales accross items
item_daily_sales = train.groupby(['item', 'date'], as_index=False)['sales'].sum()
###Plot for total sales of series
plt.figure(figsize=(10,4))
plt.plot(daily_sales)
plt.title("Daily sales")
plt.xlabel('Date')
plt.ylabel('sales')


#Further breakdown of items and stores accros year, month, days etc.
agg_year_item = pd.pivot_table(train, index='year', columns='item',
                               values='sales', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(train, index='year', columns='store',
                                values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
#Above all items and stores seem to enjoy a similar growth in sales over the years

agg_month_item = pd.pivot_table(train, index='month', columns='item',
                                values='sales', aggfunc=np.mean).values
agg_month_store = pd.pivot_table(train, index='month', columns='store',
                                 values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_month_item / agg_month_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Month")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_month_store / agg_month_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Month")
plt.ylabel("Relative Sales")
#Above all items and stores seem to share a common pattern in sales over the months as well

agg_dow_item = pd.pivot_table(train, index='dayofweek', columns='item',
                              values='sales', aggfunc=np.mean).values
agg_dow_store = pd.pivot_table(train, index='dayofweek', columns='store',
                               values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_dow_item / agg_dow_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_dow_store / agg_dow_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
#All items and stores also seem to share a common pattern in sales over the days of the week as well
######## Could continue explore different relations and degeneracies.

agg_dow_month = pd.pivot_table(train, index='dayofweek', columns='month',
                               values='sales', aggfunc=np.mean).values
agg_month_year = pd.pivot_table(train, index='month', columns='year',
                                values='sales', aggfunc=np.mean).values
agg_dow_year = pd.pivot_table(train, index='dayofweek', columns='year',
                              values='sales', aggfunc=np.mean).values

plt.figure(figsize=(18, 5))
plt.subplot(131)
plt.plot(agg_dow_month / agg_dow_month.mean(0)[np.newaxis])
plt.title("Months")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.subplot(132)
plt.plot(agg_month_year / agg_month_year.mean(0)[np.newaxis])
plt.title("Years")
plt.xlabel("Months")
plt.ylabel("Relative Sales")
plt.subplot(133)
plt.plot(agg_dow_year / agg_dow_year.mean(0)[np.newaxis])
plt.title("Years")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")

agg_store_item = pd.pivot_table(train, index='store', columns='item',
                                values='sales', aggfunc=np.mean).values

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(agg_store_item / agg_store_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Store")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_store_item.T / agg_store_item.T.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Item")
plt.ylabel("Relative Sales")
plt.show()



####OBS:  Consider why this sort of data exploration is important? There could be different dependcies, in such a case what to do?

################################## XGBoost modelling ##############################33
test = test.drop(['date','id'], axis = 1)
train = train.drop(['date'], axis = 1)

#### XGBoost model ###
#Choose length of the training set!
train_size = int(len(train) * 0.7)
#Split in train and test dataset.
train_n, test_n = train[0:train_size], train[train_size:len(train)]
print('Observations: %d' % (len(train)))
print('Training Observations: %d' % (len(train_n)))
print('Testing Observations: %d' % (len(test_n)))

train_n = pd.DataFrame(train_n)
test_n = pd.DataFrame(test_n)

#As XGBoost is a decision tree and regression mix such that y(sales) = x(parameters) in our train set we drop the sales column.
x_train = train_n.drop('sales', axis=1)
y_train = train_n.pop('sales')
x_test = test_n.drop('sales', axis=1)
y_test = test_n.pop('sales')

#Alternative approach
x_train,x_test,y_train,y_test = train_test_split(train.drop('sales',axis=1),train.pop('sales'),random_state=42,test_size=0.2)

#Model:
def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:squarederror','eval_metric':'mae'}
                  ,dtrain=matrix_train,num_boost_round=500, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
#Predit
y_pred = model.predict(xgb.DMatrix(test), ntree_limit = model.best_ntree_limit)
test1['sales'] = y_pred
print(test1)

"""
# We could also add lagges features such that it becomes a supervised ML problem.
test1['store_item_shifted_365'] = test1.groupby(["item","store"])['sales'].transform(lambda x:x.shift(365)) # sales for that 1 year  agoago
test1['store_item_shifted_365'].fillna(test1['store_item_shifted_365'].mode(), inplace=True)
store_item_shifted_365=test1.groupby(['item','store'])['sales'].shift(365).reset_index

#Adding lagged features to train data with predicted sales and rebuilding the model

train1['store_item_shifted_365'] = train1.groupby(["item","store"])['sales'].transform(lambda x:x.shift(365)) # sales for that 1 year  ago
train1['store_item_shifted_365'].fillna(train1['store_item_shifted_365'].mode(), inplace=True)
store_item_shifted_365=train1.groupby(['item','store'])['sales'].shift(365).reset_index

#Train model again:
train_size1 = int(len(train1) * 0.7)
train_n1, test_n1 = train1[0:train_size1], train1[train_size1:len(train1)]
print('Observations: %d' % (len(train1)))
print('Training Observations: %d' % (len(train_n1)))
print('Testing Observations: %d' % (len(test_n1)))

train_n1 = pd.DataFrame(train_n1)
test_n1 = pd.DataFrame(test_n1)

x_train1 = train_n1.drop('sales', axis=1)
y_train1 = train_n1.pop('sales')
x_test1 = test_n1.drop('sales', axis=1)
y_test1 = test_n1.pop('sales')


#x_train1,x_test1,y_train1,y_test1 = train_test_split(train1.drop('sales',axis=1),train1.pop('sales'),random_state=42,test_size=0.2)

# Calling the same function with new train and test sets, which contain added lagged features
model1=XGBmodel(x_train1,x_test1,y_train1,y_test1)

forecast_XG = pd.DataFrame()
y_pred = model.predict(xgb.DMatrix(test), ntree_limit = model.best_ntree_limit)

forecast_XG['sales']= y_pred
print(forecast_XG)
"""
############################# ARIMA modelling ###############################################
#Notice that I load the data again, as the feature engineering is only important in ML models
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
df_train = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/Item and Demand/train.csv', parse_dates=['date'], index_col=['date'])
df_test = pd.read_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/Item and Demand/test.csv', parse_dates=['date'], index_col=['date'])
print(df_train)


#creating empy series of sales results, such that we upload the result in this dataset. 
sarima_results = df_test.reset_index()
sarima_results['sales'] = 0
#print(sarima_results)
tic = time.time()

#This for-loop is only possible since we discovered pretty much every shop and item share the same features. If this wasn't the case we could not just for-loop with same model, and
#different model configuartions would be needed. For larger dataset application I would probably recommend using something else.

for s in sarima_results['store'].unique():
    for i in sarima_results['item'].unique():
        si = df_train.loc[(df_train['store'] == s) & (df_train['item'] == i), 'sales']
        sarima = sm.tsa.statespace.SARIMAX(si, trend='n', freq='D', enforce_invertibility=False,
                                           order=(6, 1, 0))
        results = sarima.fit()
        fcst = results.predict(start='2017-12-31', end='2018-03-31', dynamic=True)
        sarima_results.loc[(sarima_results['store'] == s) & (sarima_results['item'] == i), 'sales'] = fcst.values[1:]
        
        toc = time.time()
        if i % 10 == 0:
            print("Completed store {} item {}. Cumulative time: {:.1f}s".format(s, i, toc-tic))

print(sarima_results.head())
sarima_results.to_csv('C:/Users/shansen075/Desktop/Files/Forecast/Data for workshop/Item and Demand/forecast_timeseries.csv', index=False)
