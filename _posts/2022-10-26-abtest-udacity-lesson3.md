---
layout: post
title: A/B Testing - Lesson3 Choosing and Characterizing Metrics
date: 2022-10-26
categories: ABtest metric
comments: true
---

## Before definding metrics
* Invariant Checking : shouldn't chnage between control group and test group. e.g. user numbers
* Variation : revenue, market share, page views.. 

1. High level concept 
    * e.g Active user
2. How to define the detailed
    * how to define "active"
3. How to calculate , mean/sum..

* how generally applicable the metric is

### High leve metrics : Customer Funnel

![](https://i.imgur.com/kWCrz28.png)
![](https://i.imgur.com/eI2EMqo.png)

* 步驟不一定是線性的, 可能反覆進行
* 每一個步驟都可以是Metric
* 可以選擇特定兩步驟計算轉換率
* 如何選擇?

![](https://i.imgur.com/i5OUVjN.jpg)
![](https://i.imgur.com/okXMOo1.jpg)


## Difficult Metrics
* Don't have access to data
* Takes too long

* Whcic metric would be hard to measure?
![](https://i.imgur.com/ZG2lJpd.png)

### Other techniques
* survey
* external data


1. Retrospective analysis
* correlation analysis but not causation

2. Gather additional data

![](https://i.imgur.com/D4kNnVq.png)

* Scenario to use different techniques
![](https://i.imgur.com/TCrHQ04.png)


### Metric Definition and data capture

![](https://i.imgur.com/QduDZL5.png)

* Def #1 (Cookie probability): For each <time interval>, number of cookies that click divided by number of cookies
* Def #2 (Pageview probability): Number of pageviews with a click within <time interval> divided by number of pageviews
* Def #3 (Rate): Number of clicks divided by number of pageviews
    
![](https://i.imgur.com/uZW3k8y.png)

### Filtering and segmentation
  * de-bias : filter out / find abnormal
  * external reasons such competitors, website rebranding
  * filter spams and frauds
  * set up baseline  
    
  * Example
    ![](https://i.imgur.com/UxVLdja.png)
    ![](https://i.imgur.com/sNEZNX4.png)
    ![](https://i.imgur.com/Z5d9EyY.png)
    ![](https://i.imgur.com/m7SsWdB.png)

  * Example
    
    ![](https://i.imgur.com/CEOjixe.png)
![](https://i.imgur.com/Chzc1EH.png)
![](https://i.imgur.com/NMmWyQU.png)
    ![](https://i.imgur.com/BKVxA2b.png)
![](https://i.imgur.com/6l7qmaQ.png)

![](https://i.imgur.com/c2e4W0T.png)


## Summary Metrics -sumarize all the individual metrics
    
1. count/sum
2. distribution metrics
3. passibilities/rate
4. ratio
* How to choose? -> 
    1. Sensitivity , robustness
    2. how the distribution looks like
      * X: all the distinct value/Y: frequency
        * normal -> median maybe
### Common distributions in online data
> measure the average staytime on the results page before traveling to a result.->  您可能會看到我們所說的 Poisson distribution，或者停留時間呈 exponentially distributed。
> Another common distribution of user data is a “power-law,” Zipfian or Pareto distribution.
    
> To see how probabilities are really averages, consider what the probability of a single user would look like - either 1 / 1 if they click, or 0/1 if they don't. Then the probability of all users is the average of these individual probabilities. For example, if you have 5 users, and 3 of them click, the overall probability is (0/1 + 0/1 + 1/1 + 1/1 + 1/1) divided by 5. This is the same as dividing the total number of users who clicked by the number of users, but makes it more clear that the probability is an average.
    
![](https://i.imgur.com/a72DDM2.png)

 ![](https://i.imgur.com/rFsQTAB.png)

### Sensibility and Robustness
* mean - sensibility
* median - robustness
* How to measure?
    * conduct some experiments
    * Retrospective cohort study
    
* Example : the latency of a video
    1. Retrospective analysis to focus on stable information
![](https://i.imgur.com/cneiG74.png)
![](https://i.imgur.com/RR7FhMu.png)

    * not be too much difference between different videos for a good metric if the vidoe are similar.
    
    ![](https://i.imgur.com/smsxBS7.png)
90th and 99th percentile : not robust enough
    
    2. Run a new experiment to focus on changes
    * video 1 has the highest resolution.
    ![](https://i.imgur.com/MdUQq1x.png)
    ![](https://i.imgur.com/JvLKanj.png)

