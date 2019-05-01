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
**Model 1--Hand-crafted Logistic Regression w/ adagrad optimizer \[added "age_squared" and "education yrs"\]**
 * Public score: 85.552 % 
 * Private score: 85.468 %
    
**Model 2--Scikit-learn Linear SVC w/ C=20.0 \[used same features as in Model 1\]**
 * Public score: 85.577 % 
 * Private score: 85.456 %
   
**Model 3--Scikit-learn Gradient Boosting Classifier w/ depth=6, 250 estimators \[applied parameter tuning\]**
 * Public score: 87.383 % 
 * Private score: 87.485 % _(Final Rank: Top 15%, **24/165**)_
