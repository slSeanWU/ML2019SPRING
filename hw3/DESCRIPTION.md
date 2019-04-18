# HW3: Convolutional Neural Network
## I. Task
 Given labeled training data (grayscale face images), classify the **Facial Expression** into the following 7 categories
 * 0: Angry   
 * 1: Disgust   
 * 2: Fear  
 * 3: Happy    
 * 4: Sad    
 * 5: Surprise    
 * 6: Neutral    
## II. Data
 * FER2013 Dataset
 > 48\*48 Non-aligned grayscale face images _(28709 for training, 7178 for competition)_
 
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
