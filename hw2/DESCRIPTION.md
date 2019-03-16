# HW2: Binary Classification -- Income >50K ?
## I. Task
 Given labeled training data consisting of personal attributes (_including **age**, **education**, **weekly working hours**...etc._), try to come up with a **Binary Classifier** which predicts whether a person earns _**>50K**_.
## II. Data
 * .csv files
 > Raw data containing categorical attributes _(e.g. marital status, highest degree...)_, as well as continuous attributes
 * X_train, y_train, X_test
 > Extracted data with **one-hot** encoding for categorical features
 
## III. Result
**Error Measure: Categorical Accuracy**  
> Kaggle in-class competition results   
>  
**Model 1--Hand-crafted GD w/ adagrad optimizer \[all features linear\]**
 * Public score: 5.69152  
 * Private score: 7.21953
    
**Model 2--Scikit-learn Ridge Regression w/ lambda=0.03 \[feature engineering applied\]**
 * Public score: 5.51633  
 * Private score: 7.01786 _(Final Rank: Top 25%, **43/176**)_
