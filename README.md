
# [Project 1: Shiny Web App in R](https://github.com/MarkMData/Shiny_web_app_project/tree/main)  

## Overview

This was a project I completed as part of an R programming module from an MSc in Data Analytics  

This aim of this project was to build a Shiny app which visualises and summarises hourly metorological data collected at 20 locations across the United Kingdom between 1st January to 30th November 2021. The data was obtained from the Met Office Integrated Data Archive System and provided to us in csv format.
  
The app was required to have the following functionality:
- Allow the user to select up to five locations which should then be displayed on a map
- Allow the user to select one of four weather variables
- Allow the user to select the aggregation level of the weather variable data to be displayed
- Allow the user to select the time period to display
- Produce time series plots for selected variables, locations, and aggregation level.
- Display a table of the daily means for all weather variables for selected sites
- Produce a downloadable report of plots, table, and map for selected sites and weather variable
- Produce a downloadable csv file of the table
- Calculate and display the Hutton Criteria (a diagnostic which is used to alert farmers of the risk of potato blight forming on potato crops) for a selected location and month

## Approach  
With a lot of infomation to display I decided to go with a two page layout using the flatly theme. The first page is used for displaying the meteorological data.  

  
![Image of Shiny web app page 1](https://github.com/MarkMData/portfolio/blob/main/images/Shiny_app_pg1.PNG?raw=true)  
The the second page is used for displaying the Hutton Criteria.  

![Image of Shiny web app page 2](https://github.com/MarkMData/portfolio/blob/main/images/Shiny_app_pg2.PNG?raw=true)  

To create the app I used a single app.R file for the UI and server code, and a seperate R file for all the functions used for generating the plots, maps, and table. I also created a Rmarkdown file for the downloadable report.  

## Data cleaning and wrangling using Tidyverse packages

The data sets for each location were provided to us as csv files and loading these into the app was slow, so I converted them all to RData files. I created functions for generating the plots, table, and Hutton Criteria calender and these incorporated the following actions:
- Loading the Rdata files for the selected locations (up to 5) and store them in a list of dataframes
- Removing observations for the 29th Febuary which did not exist in 2021
- Creating a date_time variable from year, month, day and hour data
- Joining the location dataframes with a dataframe containing the complete number of hours for the entire period. This was required as some locations were missing rows.
- Removing duplicate observations
- Performing aggregations on the selected variable if required
- Creating new variables to indicate times when temperature and humidity were below a threshold, and to indicate when the Hutton Criteria had been met
- Generate plots/table based on user selections

## Visualisations  
- I chose to display the location map in a tab on the first page, and as the background of the second page. To create the maps I used the Leaflet package, and choose a base map with neutral colours and no text, to ensure the selected locations would be clearly visable. Locations on the maps are indicated with markers, postioned using longitude and latitude values.
- The plots were created using the GGplot2 package.
- I used the CalendR package to create a simple clean monthly calender to display the days the Hutton Criteria was met.

---  

# [Project 2: Deep Learning with Tensor Flow](https://github.com/MarkMData/TensoFlow_project/tree/main)  
## Human activity classification  
<br>  

## Overview  
The aim of this project was the classification of of human activity with neural networks using TensorFlow. The data was obtained from the <ahref="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones" target="_blank"> UCI Machine Learning Repository</a>  and was created by Reyes-Ortiz, Anguita, Ghio, Oneto, & Parra (2012). The data set contains of readings from the accelerometer and gyroscope contained within a smartphone that was worn by 30 volunteers as they were performing six activities (walking, walking upstairs, walking downstairs, sitting, standing and laying). All the data had been collected at 50Hz and consists of triaxial linear acceleration data (that was separated into body acceleration and gravity components) and triaxial angular velocity data. There were 10299 total samples of 128 readings, each representing 2.56 seconds of activity and these had been randomly split into a training set with 7352 observations (70%) and a testing set with 2947 observations (30%). I split the training set again into training (80%) and validataion (20%) sets for use during model development.  
<br>  
<br>
The goal of the analysis was to evaluate the accuracy of three different neural network models (of increasing complexity) for predicting the activity class from the acceleration and velocity time segments using Python, TensorFlow, and Keras to develop the models, Tensorboard to visualise the progression of training over time.  
  
<br>  

## Model development  

<br>  

### Model 1.  

The first neural network approach consisted of a single layer implementing a multinomial regression model using only the body acceleration data and was implemented as follows:
- The individual x, y and z axis components of the body acceleration data were combined and flattened to form one-dimensional arrays.
- The activity labels for the training and testing data sets had been coded numerically and these were converted to one hot encoding format.
- The body acceleration and activity labels training data sets were then split to create training (80% of observations) and validation (20% of observations) sets, with the training set being used to develop models and the validation set being used to evaluate them.
- The multinomial regression model consisted of a SoftMax activation function, for creating classification probabilities, combined with a cross entropy loss function to minimize training error.
- I used an Adam optimizer to minimize the loss function as it can deal with noisy gradients, has an adaptive learning rate and has been shown to consistently perform well compared to other optimizers (Kingma & Lei Ba, 2015).
- After some experimentation with different learning rates, a value of 0.001 was chosen as a balance between time to convergence and accuracy & stability and at this value running for 20,000 epochs was sufficient for the loss, training accuracy and validation accuracy to stabilize without over fitting (see Figure 1).
<br>

![Figure 1.](https://github.com/MarkMData/portfolio/blob/main/images/Tf_proj_image1.png?raw=true)  
***Figure 1. Training loss, training accuracy, and validation accuracy for Model 1.***  
<br>
Accuracy for model 1 using the training data was 0.406 and for the validation data it was 0.292. This represents poor performance and could be due to two factors. Firstly, only the body acceleration data was used, and secondly by flattening the independent x, y and z components into a single one-dimensional array information about the inter-axis relationships is lost.  

### Model 2.  

The second neural network approach used was a one-dimensional convolutional neural network, built using the Keras API, that took the combined x, y and z body acceleration data with a (128, 3) shape as input and had the following five layers arranged sequentially:
1. One-dimensional convolution layer with a kernel size of 4 and 32 filters.
2. Batch normalization layer.
3. ReLu activation layer.
4. Global average pooling layer.
5. Dense layer with SoftMax activation.

This was implemented in the following way:
- An Adam optimizer was used for this approach for the reasons described previously.
- The learning rate was set at 0.0001 as at values higher the training process was unstable.
- The model was run using three different batch sizes 32, 64 and 128 for 2000 epochs which was long enough for the accuracy of both the training and validation data to plateau without overfitting.
- The loss and accuracy for both the training and validation data at the three different batch sizes were monitored using Tensorboard (see Figure 2) 
- The validation accuracy was recorded after each epoch, and the parameters associated with the greatest accuracy saved. (the highest validation accuracy, along with loss and
training accuracy, for each batch size are displayed in Table 1).
<br>

![Figure 2.](https://github.com/MarkMData/portfolio/blob/main/images/Tf_proj_image2.png?raw=true)  
***Figure 2. Accuracy and loss for model 2 with three different batch sizes. Training data is blue and validation data is red.***  
<br>  

For all three batch sizes the highest validation accuracy was above 0.85, indicating quite good predictive performance, and was slightly higher than the training accuracy indicating no issues with overfitting. The stability of the training and validation accuracy and loss improved as batch size increased, but time to convergence was slower. The best overall accuracy on the validation set of 0.871 was achieved using the medium batch size of 64.
<br>  

    
***Table 1. Accuracy and loss for model 2 with different batch sizes***
|                     | Batch size = 32 | Batch size = 64 | Batch size 128 |
|---------------------|-----------------|-----------------|----------------|
| Training loss       |     0.425       |     0.439       |     0.463      |
| Validation loss     |     0.440       |     0.455       |     0.479      |
| Training accuracy   |     0.854       |     0.857       |     0.847      |
| Validation accuracy |     0.867       |     0.871       |     0.861      |  

<br>

With the learning rate set at 0.0001, the batch size set to 64 and running for 2000 epochs several model parameter configurations were then trialed with the aim of finding the combination that produced the best validation accuracy. Initially, three different kernel sizes (3, 4, 5) and three different options for the number of filters (16, 32, 64) were used with a single convolutional layer. As the number of filters or size of kernel increased, the time to convergence became longer and while some marginally greater improvement in accuracy may have been achieved with running for greater than 2000 epochs this was not practical given the time and resources available. A kernel size of three had worse validation accuracy than the other two sizes at all numbers of filters, and kernel size five was better than four when used with 16 and 32 filters but worse at 64 filters. Validation accuracy improved with the number of filters for all kernel sizes.  

While keeping the learning rate, batch size and number of epochs the same, two models were also tried with additional convolutional and batch normalization layers that had the following configurations.  

**Two convolutional and two batch normalization layers:**
1. One-dimensional convolution layer with a kernel size of 3 and 32 filters.
2. Batch normalization layer
3. One-dimensional convolution layer with a kernel size of 4 and 64 filters.
4. Batch normalization layer
5. ReLu activation layer
6. Global average pooling layer
7. Dense layer with SoftMax activation

**Three convolutional and three batch normalization layers:**
1. One-dimensional convolution layer with a kernel size of 3 and 16 filters.
2. Batch normalization layer
3. One-dimensional convolution layer with a kernel size of 4 and 32 filters.
4. Batch normalization layer
5. One-dimensional convolution layer with a kernel size of 5 and 64 filters.
6. Batch normalization layer
7. ReLu activation layer
8. Global average pooling layer
9. Dense layer with SoftMax activation

The accuracy values for all iterations of model 2 are displayed in Table 2. Compared to the best performing model with a single convolutional layer, the two models with multiple convolutional layers had lower validation accuracy and suffered from greater instability during training, as illustrated in Figure 4.  
<br>
***Table 2. Accuracy of model 2 with different parameter configurations***
|     Model variations                                                                                                               |     Training accuracy    |     Validation accuracy    |
|------------------------------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------|
|     1 convolutional layer, kernel size = 3, filters = 16                                                                           |     0.824                |     0.838                  |
|     1 convolutional layer, kernel size = 4, filters = 16                                                                           |     0.833                |     0.847                  |
|     1 convolutional layer, kernel size = 5, filters = 16                                                                           |     0.845                |     0.859                  |
|     1 convolutional layer, kernel size = 3, filters = 32                                                                           |     0.847                |     0.857                  |
|     1 convolutional layer, kernel size = 4, filters = 32                                                                           |     0.857                |     0.871                  |
|     1 convolutional layer, kernel size = 5, filters = 32                                                                           |     0.864                |     0.873                  |
|     1 convolutional layer, kernel size = 3, filters = 64                                                                           |     0.854                |     0.868                  |
|     1 convolutional layer, kernel size = 4, filters = 64                                                                           |     0.867                |     0.880                  |
|     1 convolutional layer, kernel size = 5, filters = 64                                                                           |     0.863                |     0.878                  |
|     2 convolutional layers, (kernel size = 3, filters = 32), (kernel size   = 4, filters = 64)                                     |     0.875                |     0.876                  |
|     3 convolutional layers, (kernel size = 3, filters = 16), (kernel size   = 4, filters = 32), (kernel size = 5, filters = 64)    |     0.878                |     0.876                  |  
<br>
<br>
  
![Figure 3.](https://github.com/MarkMData/portfolio/blob/main/images/Tf_proj_image3.png?raw=true)  
***Figure 3. Training and validation accuracy for the best performing model with a single convolutional layer (left) versus the network with two convolutional layers (right). Training data is blue and validation data is red***
<br>
### Model 3.
The final network approach (network 3) involved using all nine input variables, which were the x, y, z components for each of the body acceleration, gravity and angular velocity data.
- The nine inputs were combined into an array of shape (128, 9) and then the training set was split into training (80% of observations) and validation (20% of
observations) sets.
- This data was then used with two network configurations:
    1. The model 1 configuration that had produced the best validation accuracy (single convolutional layer with kernel size 4 and 64 filters).
    2. The model 2 configuration that had included three convolutional layers.
- For both configurations the learning rate was kept at 0.0001, the batch size at 64 and the number of epochs 2000.
<br>
***Table 3. Training and validation accuracy and loss for the two configurations of model 3.***

|                            |     Single convolutional layer    |     Three convolutional layers    |
|----------------------------|-----------------------------------|-----------------------------------|
|     Training loss          |     0.046                         |     0.032                         |
|     Validation loss        |     0.057                         |     0.039                         |
|     Training accuracy      |     0.985                         |     0.994                         |
|     Validation accuracy    |     0.982                         |     0.991                         |  

<br>
Both configurations had excellent validation accuracy (see Table 3) with the single convolutional layer network achieving 0.982, and the network with three convolutional layers achieving 0.991. Looking at the plots of loss and accuracy during training (see Figure 4) both configurations of network 3 exhibited good stability during training.  
<br>
<br>  

![Figure 4.](https://github.com/MarkMData/portfolio/blob/main/images/Tf_proj_image3.png?raw=true)  
***Figure 4. Accuracy and loss for model 3. The single convolutional layer configuration is on the right and the three convolutional layer configuration is on the left. Training data is blue and validation data is red.***  
<br>  
## Performance of models on test data.  
Finally, the best performing configurations for network 1, network 2 and network 3 were evaluated against the test data (see Table 4 for results). The test accuracy for model 1 was 0.307 which is slightly better than the model achieved on the validation data but still poor classification performance. The best performing configuration for model 2 (single convolutional layer with kernel size 4 and 64 filters) had accuracy on the test data of 0.843 which is quite good performance and only slightly less than what was achieved on the validation data. The configuration of model 3 with three
convolutional layers had accuracy against the test data of 0.958 which represents excellent performance and only slightly lower than the validation data.  

<br>  

***Table 4. Accuracy of all three models on testing data***  

|         | Test accuracy |
|---------|---------------|
| Model 1 | 0.307         |
| Model 2 | 0.843         |
| Model 3 | 0.958         | 

 <br>  

The convolutional networks had the advantage over the multinomial regression network of using the input data in a shape that allowed the spatial relationships between the x, y and z axis to be preserved. They also had multiple layers in which features within the data could be identified and then combined.  
<br>  
### References  
Kingma, D. P., & Lei Ba, J. (2015). Adam: A method for stochastic optimization. ICLR, (p. 1:15).  

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2012). Human Activity Recognition Using Smartphones. Retrieved from UCI Machine Learning Repository: ttps://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones  
<br>  

# [Project 3: Machine Learning Classification using R](https://github.com/MarkMData/ML_classification_project/tree/main)  

## Predicting drug use with five machine learning algorithms  
<br>  

***Note: This project was completed as part of an MSc in Data Analytics and uses a modified version of the data used in the study by Fehrman et al., (2017). The number of observations have been reduced and the data has been externally scaled to change some categorical variables to pseudo-continuous variables in order to simplify the analysis. It was provided to us in this format***  

<br>  

## Overview  

The aim of the project was to evaluate whether drug use could be predicted from personality trait and demographic variables using five classification methods (logistic regression with lasso penalty, k-nearest neighbours, classification trees, random forests and support vector machines).  The classification threshold used for all models was 0.5 such that probabilities above the threshold were deemed to indicate drug use and values below the threshold no drug use and the metrics used to evaluate model performance were accuracy, sensitivity and specificity. All analysis was completed using R.  

<br>  

## Exploratory data analysis and pre-processing  
The data set comprised of 600 observations with the dependent variable being a binary classifier indicating whether an individual had ever taken legal or illegal drugs (never taken drugs n = 300, has taken drugs n= 300) and the 11 independent variables all continuous (see Table 1 for an overview of the variables).  The dataset had no missing values.  

<br>  

***Table 1. Independent variables used in the analysis***  

|     Variable     |     Description                                                              |
|------------------|------------------------------------------------------------------------------|
|     Age          |     Age of study participants                                                |
|     Education    |     Education level of study participants                                    |
|     X.country    |     Country of origin for study participants                                 |
|     Ethnicity    |     Ethnicity of study participants                                          |
|     Nscore       |     Participants neuroticism personality trait   score                       |
|     Escore       |     Participants extraversion personality trait   score                      |
|     Oscore       |     Participants openness to experience   personality trait score            |
|     Ascore       |     Participants agreeableness personality trait   score                     |
|     Cscore       |     Participants conscientiousness personality   trait score                 |
|     Impulsive    |     Participants impulsiveness personality trait   score                     |
|     SS           |     Participants sensation personality trait   seeking score                 |
|     Class        |     Whether participants had ever taken legal or   illegal drugs (yes/no)    |  
<br>  

The dataset was split into training (50% of observations), validation (30% of observations) and test (20% of observations) sets while preserving the distribution of the response variable, and this was done using the caret package. The training dataset was used for the exploratory analysis and model building, the validation dataset was used for comparing the performance of the different modelling approaches and the test dataset was used to assess the performance of the model that had the highest prediction accuracy on the validation data. The relationships between the response variable and the predictors are displayed in the boxplots of Figure 1. From the plots there appears to be an association between drug use and the variables Age, SS and X.Country, and several variables appear to have outliers.  

<br>  

![Figure 1](https://github.com/MarkMData/portfolio/blob/main/images/boxplots.jpeg?raw=true)  

***Figure 1. Distributions for predictor variables by drug use (0 = never used, 1 = has used).***  
<br>  

Figure 2 contains correlations, scatterplots, and density plots for all the independent variables. There are no strong correlations between any of the predictor variables meaning multicollinearity was not a concern, but there is evidence that some are not normally distributed and there appears to be an extreme outlier in the ethnicity data.  
<br>  
![Figure 2](https://github.com/MarkMData/portfolio/blob/main/images/pairsplot.jpeg?raw=true)  
***Figure 2. Relationships between predictor variables with drug use as a class (0 = never used, 1 = has used).***  
<br>  
From the boxplots and scatter plots it appeared that the Ethicity variable was comprised of very few unique values. To identify if the Ethnicity (or any other variables) had a near zero variance the percentage of unique values and the frequency ratio of the most prevalant to the second most prevalant value for each variable was calculated (using the caret package), with cut offs of 10% for the unique values and 20:1 for the frequency ratio, as recommended by Kuhn & Johnson (2013). The only variable to meet both criteria was the Ethicity variable with a frequecy ratio of 30.44 and only 2.33% unique values. As the variable Ethnicity had near zero variance, and contained an extreme outlier, it was excluded from all the models. As some of the classification methods such as k-nearest neighbours are sensitive to differences in scale of the variables, the training, validation and test dataset were centred and scaled using the mean and standard deviation from the training data before being used in modelling, and this was also done using the caret package.  
<br>  

## Method and Results  
<br>  

### Logistic regression with lasso penalty model  
The logistic regression model with a lasso penalty was fit using the glmnet package. 100 values of lambda were evaluated using 10-fold cross validation to identify when the minimum misclassification error occurred and then the value within one standard error of this (lambda = 0.196) was selected (see Figure 3).  
<br>  
![Figure 3](https://github.com/MarkMData/portfolio/blob/main/images/lassoCV.jpeg?raw=true)  
***Figure 3. Logistic regression with lasso penalty model cross validation miss-classification error for different values of log lambda. Vertical lines are placed at the minimum CV error (left) and one standard error from the minimum (right).***  

<br>  
With the selected value of lambda, all but two of the variable coefficients were shrunk to zero, Age and X.Country, and both of these had negative values of -0.228 and -0.085 respectively. On the validation data the penalised logistic regression model had sensitivity of 0.811, specificity of 0.678 and accuracy of 0.744 (results are displayed in Table 4.).  

<br>  

### K-nearest neighbour’s model  
To identify the optimal value for k, 10-fold cross validation was used to iteratively assess the prediction accuracy for 50 values of k ranging from one to 99 (odd values only to prevent ties) with the best prediction accuracy occurring at k = 97 (see Figure 4). When tested against the validation data the KNN model with k = 97 had sensitivity of 0.833, specificity of 0.822 and accuracy of 0.828 (results are displayed in Table 4).  

<br>  

![Figure 4](https://github.com/MarkMData/portfolio/blob/main/images/knnplot.jpeg?raw=true)  
***Figure 4. 10-fold cross validation accuracy for KNN model with odd values of k from 1 and 99.***  

<br>  

### Classification tree model  

A full classification tree was created (using the rpart package) and then pruned, to reduce the likelihood of overfitting, based on the complexity parameter that corresponded to the lowest average prediction error within one standard deviation from the minimum prediction error, determined by 10-fold cross validation. This resulted in a small tree with only the variables X.Country and Age included (see Figure 5). Sensitivity, specificity and accuracy for the classification tree against the validation data were 0.811, 0.7 and 0.7889 respectively (see Table 4).
<br>  

![Figure 4](https://github.com/MarkMData/portfolio/blob/main/images/treePlot.jpeg?raw=true)  
***Figure 5. Classification tree after pruning.***  

<br>  

### Random forests model  
The random forest model involved constructing many trees using bootstrapped samples of the training data and limiting the variables selected for each split to a random subset of the full variable set, and then averaging the result. To identify the best number of variables to include for random selection at each split, 10-fold cross validation was used to compare values from two to the full number of predictors. The best average prediction accuray was obtained when including 6 predictors for selection at each split and the resulting performance on the validation data with this configuration was sensitivity of 0.767, specificity of 0.733 and accuracy of 0.75 (results in table 4).  

<br>  

![Figure 5](https://github.com/MarkMData/portfolio/blob/main/images/forestplot.jpeg?raw=true)  
***Figure 5. Average 10-fold cross validation accuracy for the random forest model with different numbers of predictors included at each split.***  
<br>  
### Support vector machines model  
The e1071 package was used to implement a SVM model with radial basis function. A grid of values for the cost parameter (2 raised to the power of integers from -2 to 10) and gamma (10 raised to the power of integers from -7 to 0) were evaluated using 10-fold cross validation. The average cross validation error is displayed in Figure 6, with the best tune occurring with a cost parameter = 512 and gamma = 0.00001. The performance of the best tuned SVM model against the validation data resulted in a sensitivity of 0.844, specificity of 0.733, and accuracy of 0.789 (see Table 4 for results).  
<br>  
![Figure 6](https://github.com/MarkMData/portfolio/blob/main/images/svmplot.jpeg?raw=true)  
***Figure 6. Average 10-fold cross validation error for the SVM model with different values of the cost parameter and gamma.***  
<br>  
### Summary of performance of all models against validation data  
The performance of the five models against the validation data is presented in Table 4. The highest sensitivity was achieved with the KNN and SVM models equally at (0.844), with the random forest model having the worst sensitivity (0.767). The best specificity was achieved using the KNN model (0.811) with the logistic regression with lasso penalty model performing the worst (0.678). The best overall accuracy was achieved using the KNN model (0.828) with the logistic regression with lasso penalty model having the lowest accuarcy (0.744).  
<br>


***Table 4. Sensitivity, specificity and accuracy of all models against the validation data***
|             | Logistic regression  with L1 penalty | KNN         | Trees       | Random forests | Support vector machines |
|-------------|--------------------------------------|-------------|-------------|----------------|-------------------------|
| Sensitivity | 0.811                                | 0.844       | 0.811       | 0.767          | 0.844                   |
| Specificity | 0.678                                | 0.811       | 0.7         | 0.733          | 0.733                   |
| Accuracy    | 0.744                                | 0.828       | 0.756       | 0.75           | 0.789                   |  
<br>  

## Performance of best model on test data  

As the KNN model had the best performance of the bunch, it was used against the test dataset and had sensitivity of 0.85, specificity of 0.85, and overall accuracy of 0.85 which indicates reasonable predictive performance.  
<br>


### References  
Fehrman, E., Muhammad, A. K., Mirkes, E. M., Egan, V., & Gorban, A. N. (2017). The Five Factor Model of personality and evaluation of drug consumption risk (arXiv:1506.06297). arXiv. https://doi.org/10.48550/arXiv.1506.06297  
Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer. https://doi.org/10.1007/978-1-4614-6849-3

