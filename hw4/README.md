# HW4: Explainable AI
## I. Task
Using the following techniques and some toolkits to **observe** and **explain** the CNN classification model trained in HW3
 * Plot **Saliency Maps**
 * Visualize **Convolution Filters**
 * Find 'Pros' and 'Cons' with **LIME Image Explainer**
 
## II. Saliency Maps
 * Drawn with the help of `keras-vis` module
 * **Goal**: Find out which pixels affect the classification scores the most when altered slightly
 * Mathematical explanation: 
     - Given image **_I<sub>0</sub>_** and classification score **_S<sub>c</sub>_** of class **_c_**, compute:  
     - ![](Equations/eq_1.gif)
     - using **_Guided Backpropagation_**
 
## III. Convolution Filters
 * **Goal**: Use **_Gradient Ascent_** to obtain the input picture that produces highest level of activation for a filter 
 * Mathematical explanation:
     - _Level of Activation_: 
     - ![](Equations/eq_2.png)
     - 40 Iterations, starting with image of random noise:
     - ![](Equations/grad_asc.png)
     - applied **_L2 Normalization_** when computing gradients
 
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
 
## V. References
 * Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
     - https://arxiv.org/pdf/1312.6034.pdf
 * Gradient Ascent for Filter Visualization
     - _page 78~79_, https://www.csie.ntu.edu.tw/~yvchen/f105-adl/doc/161103_ConvolutionalNN.pdf 
