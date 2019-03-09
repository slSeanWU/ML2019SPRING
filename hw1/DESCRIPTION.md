# HW1: Linear Regression -- PM2.5 Prediction
## I. Task
 Given real-world data (_including **meteorological factors** and **pollutant concentrations**_) observed hourly for the whole year at Fengyuan, Taiwan, try to predict **future PM2.5 concentration** using **Linear Regression**
## II. Data
 .csv file
## III. Result
**Error Measure: Root-mean-square Error(RMSE)**  
> Kaggle in-class competition results   
>  
**Model 1--Hand-crafted GD w/ adagrad optimizer \[all features linear\]**
 * Public score: 5.69152  
 * Private score: 7.21953
    
**Model 2--Scikit-learn Ridge Regression w/ lambda=0.03 \[feature engineering applied\]**
 * Public score: 5.51633  
 * Private score: 7.01786 _(Final Rank: Top 25%, **43/176**)_
