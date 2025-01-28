# tcs_stock_price_prediction
# Application Overview

Objective

The proposed business application aims to empower traders and investors in their
decision-making regarding Tata Consultancy Services (TCS) stock by delivering accurate,
data-driven market trend predictions utilizing machine learning models.

Business Relevance

In the dynamic stock market, informed decisions are paramount to success. This AI-powered
forecasting tool addresses the need for timely, reliable, and actionable insights within the
context of TCS trading.

Goals

Our goals in this project were to harness machine learning techniques to analyze historical
stock data and pertinent financial indicators. By using varying models (Linear Regression,
Random Forest, Time Series, and Neural Network LSTM) we can predict future stock prices and
trends across various time horizons giving our users a tactical advantage. We will provide a user
friendly dashboard with clear visualizations and numerical summaries of both the current and
predicted market trends. We envision our project as one to democratize access to machine
learning analytics to enable investors of all experience levels.

# Data Management

Dataset

Keeping the objectives and aim of this project in mind, a dataset consisting of national stock
exchange information on Indian IT companies was selected. The dataset is publicly available on
Kaggle and consists of data on two companies: TCS and Infosys.
For the scope of this class and time limitations, we worked mainly with the TCS dataset. The
dataset consists of recordings from 1/1/2025 to 12/31/2015 timeline.

Software and Tools

The project is completed using Google Collaboratory enabling the utilization of Python and its
various libraries, TensorFlow, Keras, and GPU power to handle model training.
The TCS stock dataset consists of a total of 248 rows and 15 columns, refer to Figure 1.

Detailed explanation of these columns is as follows:
1. Date: Data on which the data is recorded
2. Symbol: NSE symbol of the stock
3. Series: Series of the stock
4. Prev Close: Last day close point
5. Open: Current day open point
6. High: Current day highest point
7. Low: Current day lowest point
8. Last: the final quoted trading price for a stock during the most recent day of trading
9. Close: Closing point of the current day
10. VWAP: volume-weighted average price is the ratio of the value traded to total volume
traded over a particular time horizon
11. Volume: the amount of a security that was traded during a given period. For every buyer,
there is a seller, and each transaction contributes to the count of total volume.
12. Turnover: Total Turnover of the stock till that day
13. Trades: Number of buy or Sell of the stock.
14. Deliverable Volume: The number of shares that move from one set of people (who had
those shares in their demat account before today and are selling today) to another set of
people (who have purchased those shares and will get those shares by T+2 days in their
demat account).
15. %Deliverable: percentage deliverables of that stock

Data Preparation

Some insights are as follows:
1. Prev Close, Open, High, Low, Last, Close, VWAP:
● The mean for each of these columns lies in similar ranges and can be considered
close to each other. This indicates there is a balanced distribution among these
variables.
● The standard deviation also shows a similar trend, indicating these variables do
not vary widely with their mean.
● The min and max show a significant difference indicating there are fluctuations in
the given timeline.
2. Volume and Turnover:
● Volume has a mean of 1,172,296, indicating significant trading activity.
● Turnover has a mean of 297,748,900,000,000, which is a very large number and
indicates traded stocks had high value.
3. Trades: The mean number of trades is 66,873.61, indicating a good trading market.
4. Deliverable Volume and %Deliverable: The mean of %Deliverable is 0.6703363,
suggesting about 67% of the traded volume on average results in the actual delivery of
stocks.

# Data Processing

Data Cleaning and Manipulation

1. Missing Values: The TCS stock dataset does not contain any null values. Hence, no
records were required to be dropped or filled.
2. Data Types: Most of the columns in the dataset are numerical, specifically integer or float
values. A few of them are categorical and hence show “object” as their data type.
3. Irrelevant features: Columns such as “Symbol” and “Series” do not add any relevancy to
the dataset since these values are not unique and are repetitive. Hence these columns
were not included in any of the model training.

Data Transformation

Normalization:
The RNN model was trained on normalized data. Normalizing the dataset is important
since this brings the features or columns to a similar scale. Sometimes the model may
be sensitive to differences in range of features. This will cause the larger scale features
to dominate the learning of the model, thereby compromising the model’s objective.
Normalizing the data removes any biases, hence this can improve the model’s
performance.
There are different techniques to normalize your data, however, this dataset was
normalized using MinMaxScaler, refer Figure 8. The actual data points for “Close” were
plotted to compare their scale to the normalized “Close” column. Notice the x-axis in, the data remains the same, however, the range of the data changes, and now
all values lie between 0 and 1.

# Analysis and Modeling

Exploratory Analysis

In the case of Recurrent Neural Networks (RNNs), it is critical to investigate the data sequence.
Visualizing sequences of stock prices and volumes over time can reveal patterns that RNNs can
efficiently detect. Furthermore, knowing the impact of various hyperparameters on model
performance via sensitivity analysis is critical.
Similarly, for the ARIMA model, the model basically analyzes the future stock price prediction by
using closed prices as y value and others as x value. the time column (using the month as the
basic value) will be one of the condition values for the time series model prediction.
Each of the models will also be calculated the mse, rmse, and mape in order to compare the
accuracy for the final results.

Data Visualization

Some insights that can be concluded from graphs plotted:
1. Trades: Most of the data points in stock trades lie in the range of 25,000 and 1,00,000.
2. Turnover: Most of the data points lie in the range of 100,000,000,000k to
300,000,000,000k
3. Volume: Most of the transactions have a volume of 500,000 to 1,500,000
4. %Deliverable: Most of the data points lie above 0.5 and below 0.8 indicating most traded
volumes that result in actual delivery of stock is more than 50% and less than 80%.

Model Building

ARIMA

Data Preprocessing

The time series model for the TCS stock prices can be separated into 3 steps: 1. Identify data
type, 2. Basic model (random walk/ time series lm), and 3. ARIMA/MA model building. In order
to predict the future value, we separate the dataset by using 70% (0:174) of the data as the
period of fit, and 30% (175:238) as hold-out data. We use “Close”, the closed price, as our
focused independent variable.

Model Building & Training

In order to select the model, we need to identify the graph of ACF and PACF. The lag decreases slowly from lag 1 to lag 24 in ACF. Also, the PACF drops quickly after lag 1,
and the lag values after lag 1 are almost all lower than the 2 error bound rate. Based on these 2
results from the graph, we can identify that the dataset is a random walk.
Furthermore, after we identify the random walk dataset, we should consider whether it contains
white noise or not. In the following graph, based on the Box-test, the p-value is less than 0.05,
therefore, we can reject the null hypothesis that the dataset contains white noise. Since the
p-value is less than 0.05, it is unnecessary to use a random walk with drift model. In the random walk model, we use close (closed price) as the independent variable. Based on
the plot, the prediction value is very fit into the actual series. The mse and mape in the following
graph also provide the same conclusion. The MAPE of the random walk model is 4.03%, which means that the model has outstanding
accuracy and performance in its prediction.
After completing the basic random walk model,. The next step for our team is to make an
ARIMA model to improve accuracy and performance.
Based on the information in the graph of PACF and ACF, the value drops quickly after lag 1in
PACF and decays slowly in ACF. Therefore, we will use AR (1, 0, 0) as our model function. The RMSE is 12.489 and the MAPE is 0.37 in ARIMA (1,0,0).


RNN

Data Preprocessing

The objective of the RNN model is to predict the closing price ahead of time, in other words, the
model requires “Close” as an input to learn from historical data. In addition to normalizing the
data, refer Figure 9, the data was transformed into univariate data i.e., data that consists of
observations based on a single characteristic. The “Date” column was set as the index and the
single column, in this case, “Close” is processed to generate dependent and independent
variables.
The X_train which is the independent variable for the training set has a size of (126, 60, 1)
indicating 126 records (because 186 - 60 = 126) of 60 sequences. However, the y_train which is
the dependent variable is of size (126,).

Model Building and Training

The model consists of LSTM layers that accept a tensor with an input shape of (60, 1) and
Dense layers. The hidden layers use a ReLu activation function however because this is a
regression problem, the output layer does not have any activation function. The model uses the
Adam compiler and mean squared error as the loss function. The model does not monitor any
other metrics during training.

Training
The training set comprises 75% of the dataset i.e., 186 records. The model was trained at 20
epochs and batch size was set to 1. Figure 24. shows the model’s loss over epochs.

Insights

Time Series Model - ARIMA

Model Evaluation: we use the remaining 30% of hold out dataset as our prediction value. We use ARIMA(1,0,0) as our model to predict the rest of the value (175:248). By using
‘High’,’Low’, ‘Open’, ‘Volume’, ‘Trade’, and ‘Deliverable Volume’ as the x variable to forecast the
close price. The prediction model has a MAPE of 0.35 percent, which provides better accuracy compared to
the basic random walk model.

RNN
 
Model Evaluation: The trained model was evaluated by making predictions on the test set which
comprises 25% of the dataset i.e., 62 records. The x_test of size (62, 60, 1) indicates 62 records
of 60 sequences. The predictions returned by x_test were plotted against y_test (represented by
Val) to provide a visual comparison of how far off the values were.
The prediction model has a MAPE of 0.35 percent, which provides better accuracy compared to
the basic random walk model.

Predictive Insights

The predictions somewhat overlapped with the y_test initially however the
model was unable to make accurate predictions for the future dates. This could be due to
several reasons, however it should be noted that the trend of close price was abnormal towards
the end of the year, it dropped abruptly between October 2015 to December 2015. Since this
pattern was different from the historical data that the model was trained on, it may be unable to
predict a close price that is way off the previous data. Although the predictions did show a drop
they still could not predict below a certain close price limit. The model can be retrained on a
larger dataset (more years) to understand the close price pattern towards the end of the year.

# Concluding Insights

In conclusion, both models provide relatively considerable accuracy and performance in
predicting future stock price value. Despite the RNN model containing the issue of overlapping,
the R^2 value reaches to 0.73, which means that 73% of the variance of the dependent variable
being studied is explained by the variance of the independent variable. Both models we
used contain some challenges that influence on the final prediction: dataset size and
independent variable selection.
The TCS stock price dataset only contains 248 variables, which is relatively small to build a
predictive model. Furthermore, the dataset uses the day as the standard time value, leading to
the difficulty in building a time series model with a season trend or drift (using the month or
quarter as the standard time value).
We expect to train our models with more stock prices (quarter or annual data) in the future to
reinforce the models and solve the accuracy issues. Moreover, on the independent variable
selection side, we expect to train our models by using more relative variables, which have a
direct influence on the stock price change.
