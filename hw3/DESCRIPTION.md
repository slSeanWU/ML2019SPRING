# HW3: Convolutional Neural Network
## I. Task
 Given labeled training data (grayscale face images), classify the **Facial Expression** into the following 7 categories:
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
 
## III. Model
### Ensemble Model (Avg. Probability) of 3 CNNs
**Training Platform: Windows 10 Notebook w/ i7-7700HQ CPU, GTX1050M GPU**  
> Each network is trained with 80~85% of the training data _(via Randomized Train/Test Split)_ 
> 
**C**: Convolution(LeakyReLU activation)  
**N**: Batch-Normalization  
**P**: Max-Pooling  
**F**: Fully-Connected    
* Dropout rates: 0.2-->0.35 for Pooling layers (increasing), 0.5 for Fully-Connected layers 
* Apply "same" padding in each layer
 
**Model 1--CNPCNPCNPCNPFNF**
 * Convolution layers: (5\*5, 64)--(4\*4, 128)--(3\*3, 256)--(3\*3, 256)
 * Max-Pooling layers: (3\*3, stride 2)--(3\*3, stride 2)--(2\*2)----(2\*2)
 * Fully-Connected layers: flatten(2304)--1024--7
 * Total # of parameters: ~3.4M
    
**Model 2--CNPCNPCNPCNPFNF**
 * Convolution layers: (5\*5, 64)--(3\*3, 128)--(3\*3, 256)--(3\*3, 512)
 * Max-Pooling layers: (3\*3, stride 2)---(2\*2)--------(2\*2)-------(2\*2)
 * Fully-Connected layers: flatten(4608)--1024--7
 * Total # of parameters: ~6.3M
 
**Model 3--CNPCNPCNPCNCNPFNFNF**
 * Convolution layers: (5\*5, 64)--(3\*3, 96)--(3\*3, 128)--(3\*3, 256)-(3\*3, 256)
 * Max-Pooling layers: (3\*3, stride 2)---(2\*2)--------(2\*2)-------------------(2\*2)
 * Fully-Connected layers: flatten(2304)--1024--256--7
 * Total # of parameters: ~3.7M  

## IV. Result
> Kaggle In-class Competition
>
**Error Measure: Categorical Accuracy**  
 * Public score: 70.966 % 
 * Private score: 70.716 % _(Final Rank: Top 19%, **27/147**)_
