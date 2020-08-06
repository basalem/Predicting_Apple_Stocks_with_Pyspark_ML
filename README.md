# Time Series Prediction in Spark: 
Predicting Apple Stock Price 


Author: Mohammed Ba Salem 

Contact: basaleemm@gmail.com

LinkedIn: [https://www.linkedin.com/in/mohammed-basalem/](https://www.linkedin.com/in/mohammed-basalem/)

## Project Overview 
 This project utilized Machine Learning approaches in big data environment to construct models that can predict Apple stock adjusted close price as a form of time series data points. Various models are implemented and tuned, namely,Linear Regression, Random Forest, Decision Trees and Gradient Boosting Regressor. Moreover, the overall performance of all models was improved by adding market features and normalized the data. To compare different performance of every model, [Symmetric Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error#:~:text=where%20At%20is%20the,the%20forecast%20value%20Ft.&text=Over-forecasting:%20At%20=,=%20110%20give%20SMAPE%20=%204.76%25) was used as evaluation metric.  
## Business Problem
The project aims to experiment time series analysis to forecast the adjusted close price at different time horizons (business days only) **daily** (1 day price) prediction, **weekly** price prediction (5 days), **bi-weekly** price prediction (10 days) , **monthly** price prediction (21 days) and **quarterly** price prediction (63 days). The data used is for 21 years (1997-2018), this will help ML model to learn and generalized. 


The assumption used to forecast stock price inherited from the stochastic model that tomorrow price is more likely to be that of yesterday plus an error (random noise). In other words, stock prices that are close together in time will be more closely related than prices that are further apart. **This will help financial analysts and investors to develop a trading strategy to take a position in stock market by selling or buying**.  


The following project is developed in DataBrick Paltform by using ApacheSpark-Pyspark. A more interactive notebook can be found **[HERE](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7052157552859543/916867181931954/4143425856907931/latest.html)**. 

## Target Variable 
This is a supervised regression machine learning problem where the target variable (Adjusted close price) is a continuous time series data points as shown in figure blow. It can be seen that data is not stationary i.e. the mean and variance are not constant over time, thus I have to remove stationarity by taking the difference with 1-day lag. Now, the target variable will be the price difference which will be predicted by the models, however, after we get the predicted price difference, it will be back-transformed to obtain predicted adjusted close price. 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Daily_AdjustedPrice.PNG)

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Normalized_Moving_Average.PNG)

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Stationary_Moving_Average.PNG)



## Methodology 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Methodology.PNG)

### Feature Engineering 
For time series problem, it is important to transform raw data into a more representative ones which help machine learning algorithm identify more pattern!  

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Feature_Engineering.PNG)


### Model Implementation 
Machine learning models are characterized by their high robust and powerful computation to learn trends, patterns and perform prediction or classification. Since time series forecasting can be re-framed as a supervised learning problem, I use 4 regression machine learning models to perform the job of predicting Apple stock price. However, most machine learning models do not have the capability to extrapolate outside the training domain, thus, I have normalized the data with logarithmic function and make models to predict the price difference instead of the actual price. Then, I use the predicted difference to back transform into predicted price. 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Model_Implementation.PNG)

### Model Comparison 
Performing time series analysis means that implementing regression model. For that, there are a lot of regression evaluation metrics such as MSE, RMSE MAE, R2 and SMAPE. For this project in all models, SMAPE metric was chosen as criteria to judge.

All the 4 machine learning models were trained with 9 features where some of them were engineered and others were imported into the data. As can be seen from below figure, most of the models have close SMAPE values, however, there are some models outperformed others. 
- 1Day, 5 Days and 10 Days price forecasting: Linear Regression outperform other models. 
-  21 Days and 63 Days price forecasting: Trees outperform Linear Regression. 
- 21 Days price forecasting: Random Forest out perform Decision Tree and Gradient Boosting.   
- 63 Days price forecasting: Decision Tree outperforms Random Forest and Gradient Boosting. 
- Unfortunately, Gradient Boosting performed very poorly and computational complexity for GBR is huge in comparison to other. Therefore, I have reduced the hyperparameters in ParamGridBuilder to only two parameters. Similarly, due to its computational complexity and low performance of GBR it is not taken forward. 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/Models_Evaluations.PNG)

## Model Evaluations and Results 
The fact that the SMAPE score for linear regression shows promising outcomes, it does not conclude that it can be used for future work. The SMAPE score for random forest and linear regression is approximately the same which is an indication that Random Forest is also another candidate to be the best model. Therefore, I deem the decrease in SMAPE score to be insignificant in comparison to model characteristic and its predictive capability. 
- Unlike Random Forest, Linear Regression is easily affected by multicollinearity and tends to overfit. 
- Random Forest tends to relate important features to target variable and handles overfitting by reducing variance through creating random set of trees and aggregate their averages. 
- Random Forest can work effectively with categorical or sentiment analysis to further boost model ability to learn in case a categorical feature is added. However, Linear Regression can not handle categorical features, we have to convert categorical features into numerical values by one hot encoding which results in long computations in case we have a lot of category class.
- Random Forest used commonly in developing trading strategy by controlling node split for extreme leaf trading or whole tree trading.  

**Random Forest wins here!**
 
1 Day price predicting:

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/1Day_RF.PNG)

5 Days price predicting: 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/5Days_RF.PNG)

10 Days price predicting: 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/10Days_RF.PNG)

21 Days price predicting: 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/21Days_RF.PNG)

63 Days price predicting: 

![alt text](https://github.com/basalem/Predicting_Apple_Stocks_with_Pyspark_ML/blob/master/images/63Days_RF.PNG)

## Challenges 

1. Poor data visualization libraries, and limitation of RDD in spark to show the first 1000 rows to plot. I have to convert some data into Pandas in order to get a good plots, however, this results in having a lot of code cells.

2. Lack of APIs and libraries in pyspark, specifically for time series CrossValidation split.

3. Community edition in Databriacks limits computation resources, many times I got my cluster terminated and sometimes they log me off while working.

 4. Spark still has does not have a large community or users, thus there was not enough community resource support explaining a certain error.


**Please check above Jupyter Notebook for more details. For interactive DataBricks cloud pySpark notebook, click [HERE](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7052157552859543/916867181931954/4143425856907931/latest.html)**
