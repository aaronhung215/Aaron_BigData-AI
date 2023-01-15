---
layout: post
title: Customer Analytics and A/B Testing in Python
date: 2022-08-24
tags: datacamp, python, keras, a/b test
categories: datacamp python keras a/b test
comments: true
---
## Leeson 1  *Key Performance Indicators: Measuring Business Success*
* Process
![](https://i.imgur.com/YhIBjIy.png)


* Cohort Conversion rate
> 一個月前註冊者於註冊後28天內消費的價錢 與整體比較
```python
# Set the max registration date to be one month before today
max_reg_date = current_date - timedelta(days=28)

# Find the month 1 values
month1 = np.where((purchase_data.reg_date < max_reg_date) &
                 (purchase_data.date < purchase_data.reg_date + timedelta(days=28)),
                  purchase_data.price, 
                  np.NaN)
                 
# Update the value in the DataFrame
purchase_data['month1'] = month1

# Group the data by gender and device 
purchase_data_upd = purchase_data.groupby(by=['gender', 'device'], as_index=False) 

# Aggregate the month1 and price data 
purchase_summary = purchase_data_upd.agg(
                        {'month1': ['mean', 'median'],
                        'price': ['mean', 'median']})

# Examine the results 
print(purchase_summary)

      gender device   month1           price       
                        mean median     mean median
    0      F    and  388.205  299.0  400.748  299.0
    1      F    iOS  432.588  499.0  404.435  299.0
    2      M    and  413.706  399.0  416.237  499.0
    3      M    iOS  433.314  499.0  405.272  299.0
```


## Lesson 2 *Exploring and Visualizing Customer Behavior*
> 試用期在一個月以前已經到期者
> 試用期結束後的第二週訂閱的轉換率
> 試用期結束後的第二週後訂閱人數/   試用期結束後兩週內訂閱的人數

> 到期後的七天內轉換率
![](https://i.imgur.com/l2vutA4.png)

![](https://i.imgur.com/1GLAbmo.png)


```python
# Group the data and aggregate first_week_purchases
user_purchases = user_purchases.groupby(by=['reg_date', 'uid']).agg({'first_week_purchases': ['sum']})

# Reset the indexes
user_purchases.columns = user_purchases.columns.droplevel(level=1)
user_purchases.reset_index(inplace=True)

# Find the average number of purchases per day by first-week users
user_purchases = user_purchases.groupby(by=['reg_date']).agg({'first_week_purchases': ['mean']})
user_purchases.columns = user_purchases.columns.droplevel(level=1)
user_purchases.reset_index(inplace=True)

# Plot the results 
user_purchases.plot(x='reg_date', y='first_week_purchases')
plt.show()
```

### Seasonality
![](https://i.imgur.com/glEjx7D.png)

Smoothing
![](https://i.imgur.com/KgCblAr.png)

### Noisy detection
![](https://i.imgur.com/G4Pdo5n.png)


![](https://i.imgur.com/H0kFzuv.png)



> ### Seasonality and moving averages
> Stepping back, we will now look at the overall revenue data for our meditation app. We saw strong purchase growth in one of our products, and now we want to see if that is leading to a corresponding rise in revenue. As you may expect, revenue is very seasonal, so we want to correct for that and unlock macro trends.
> In this exercise, we will correct for weekly, monthly, and yearly seasonality and plot these over our raw data. This can reveal trends in a very powerful way.
> The revenue data is loaded for you as daily_revenue.

```python
# Compute 7_day_rev
daily_revenue['7_day_rev'] = daily_revenue.revenue.rolling(window=7,center=False).mean()
    
# Compute 28_day_rev
daily_revenue['28_day_rev'] = daily_revenue.revenue.rolling(window=28,center=False).mean()
    
# Compute 365_day_rev
daily_revenue['365_day_rev'] = daily_revenue.revenue.rolling(window=365,center=False).mean()
    
# Plot date, and revenue, along with the 3 rolling functions (in order)    
daily_revenue.plot(x='date', y=['revenue', '7_day_rev', '28_day_rev', '365_day_rev', ])
plt.show()

```

![](https://i.imgur.com/qDbpCOR.png)


>  Exponential rolling average & over/under smoothing
> In the previous exercise, we saw that our revenue is somewhat flat over time. In this exercise we will dive deeper into the data to see if we can determine why this is the case. We will look at the revenue for a single in-app purchase product we are selling to see if this potentially reveals any trends. As this will have less data then looking at our overall revenue it will be much noisier. To account for this we will smooth the data using an exponential rolling average.
> A new daily_revenue dataset has been provided for us, containing the revenue for this product.

```python
# Calculate 'small_scale'
daily_revenue['small_scale'] = daily_revenue.revenue.ewm(span=10).mean()

# Calculate 'medium_scale'
daily_revenue['medium_scale'] = daily_revenue.revenue.ewm(span=100).mean()

# Calcualte 'large_scale'
daily_revenue['large_scale'] = daily_revenue.revenue.ewm(span=500).mean()

# Plot the date, and the raw data plus the calculated averages
daily_revenue.plot(x = 'date', y =['revenue','small_scale', 'medium_scale', 'large_scale'])
plt.show()

```

![](https://i.imgur.com/7Sth3Hk.png)



### Event and release
![](https://i.imgur.com/mtsD66N.png)



![](https://i.imgur.com/RfXoISk.png)



## Lesson 3 : *The Design and Application of A/B Testing*
* Causality
* control group and test group  
![](https://i.imgur.com/RItPpOg.png)

![](https://i.imgur.com/PYww1k5.png)

![](https://i.imgur.com/zyF9vw0.png)

* Good problem for A/B testing
	* Users are impacted indivisually
	* Test the change that can directly impact their behavior
* Bad
	* difficult to untangle the impact of the test


### Initial A/B test design
* Response variable
	* KPI or related to KPI
* Randomness of experimental units

### Preparing A/B test design
* Test sensitivity 
	* random
* Data variability
	* Standard deviation
	* variability  of purchase per user
![](https://i.imgur.com/Bmv9T2f.png)

	* Baseline CVR

### *Calculating sample size*
* Null hypothesis
	* reject Null hypothesis
![](https://i.imgur.com/5xZbQi5.png)


![](https://i.imgur.com/5hasmJj.png)


```python
# Merge the demographics and purchase data to only include paywall views
purchase_data = demographics_data.merge(paywall_views, how='inner', on=['uid'])
                            
# Find the conversion rate
conversion_rate = (sum(purchase_data.purchase) / purchase_data.purchase.count())

# Desired Power: 0.95
# CL: 0.90
# Percent Lift: 0.1
p2 = conversion_rate * (1 + 0.1)
sample_size = get_sample_size(0.95, conversion_rate, p2, 0.90)
print(sample_size)

```

## Lesson 4 *Analyzing the A/B test results*
* Crucial to validate your test data
	* does the data look reasonable
	* ensure you have a random sample
* P value
	* probability if the Null hypothesis is true
	* Low p-values
![](https://i.imgur.com/8IFYhIT.png)


