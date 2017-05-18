# PML-Courseproject
---
title: "PML Course Project"
author: "Philippine Reimpell"
date: "15 Mai 2017"
output: html_document
---

# Practical Machine Learning Course Project  
##Qualitative activity recognition using random forest classification
*Goal: Predict the manner in which the subject performs the Unilateral Dumbell Biceps Curl*


###Reading in the data  
The data for this project comes from this source: http://groupware.les.inf.puc-rio.br/har. 


The training and testing data were downloaded from here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  


The data was read into R. Empty cells were classified as NA.  
```{r}

QARtraining <- read.csv("C:\\Users\\Philippine\\Documents\\pml-training.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))

QARtesting <- read.csv("C:\\Users\\Philippine\\Documents\\pml-testing.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))

```



### Exploring the dataset and first processing steps  
The first seven variables of the test and training set were removed as they do not contribute to the prediction.

```{r}
QARtraining[,c(1,2,3,4,5,6,7)]=NULL
QARtesting[,c(1,2,3,4,5,6,7)]=NULL
```




###Identifying and removing missing values  

The original data contained a large proportion of missing values. No imputation was performed since the proportion of missing values in some of variables was over 90%. The variables with large proportions of missing values corresponded to the summary statistics of the raw sensor data. Thus, the final training data set only contained data obtained from raw sensor data and no missing values.


Values were created to sum the missing values in the columns in the training and the test set

```{r}
col.na <- colSums(sapply(QARtraining, is.na))
col.na.test <- colSums(sapply(QARtesting, is.na))
```

Then I removed all variables that have NAs from the training and testing data

```{r}
QARtraining <- QARtraining[, col.na == 0 & col.na.test==0]
QARtesting <- QARtesting[, col.na==0 & col.na.test==0]
```

```{r}
mean(is.na(QARtraining))
apply(is.na(QARtraining),2, sum)
```



###The outcome variable  
  
The outcome variable "classe" was converted to a factor variable with 5 levels. These levels correspond to performing the *Unilateral Biceps Curl*:
  
- Class A - "Exactly according to specifications"
- Class B - "Throwing the elbows to the front"
- Class C - "Lifting only halfway"
- Class D - "Lowering only halfway"
- Class E - "Throwing the hips to the front"



```{r}
QARtraining$classe <- as.factor(QARtraining$classe)
class(QARtraining$classe)
```

Further, all variables (except the outcome variable) were converted to numeric since the caret package assumes all data to be processed to be numeric.
```{r}

QARtraining[,c(1:52)] <- sapply(QARtraining[,1:52], as.numeric)

QARtesting[,c(1:53)] <- sapply(QARtesting[,1:53], as.numeric)

```


###Building the model  
To predict the "classe" based on the recorded raw sensor data, a random forest model was fitted.
The authors of the original paper also used a random forest model and justified their choice with reference to the characteristic noise in sensor data. This algorithm is characterized by a susbset of features selected in a random and independent manner with the same distribution for each of the trees in the forest (http://groupware.les.inf.puc-rio.br/har).
  
Following the advice from this great guide to building random forest models with caret, I used parallel processing (http://topepo.github.io/caret/parallel-processing.html).

```{r}


library(parallel)
library(doParallel)
library(caret)

cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)
```

I repeated the cross-validation performed in the original paper (http://groupware.les.inf.puc-rio.br/har) where the classifier was tested with 10 folds.

In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k−1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling (see below) is that all observations are used for both training and validation, and each observation is used for validation exactly once. 10-fold cross-validation is commonly used,[6] but in general k remains an unfixed parameter.(https://en.wikipedia.org/wiki/Cross-validation_(statistics))
```{r}
fitControl <- trainControl(method = "cv", number=10, allowParallel=TRUE)
```

Fit the model:
```{r}
library(randomForest)
model <- train(classe~., data=QARtraining, method="rf", trControl=fitControl)
stopCluster(cluster)
registerDoSEQ()
pred <- predict(model, newdata = QARtesting)

```


```{r}
model$results
```
The fitted model predicts the class of movement with an accuracy of over 0.99. The out-of-sample accuracy denoted by kappa is slightly lower, which also corresponds to a slightly higher out of sample error rate.

```{r}
model$finalModel
```` 


#Conclusion  
The random forest model predicts the versions of incorrect execution as well as correct execution of the Unilateral Dumbell Biceps curl with good accuracy. It is however questionable whether this approach would scale well, given that there are a large number of additional wrong executions of the exercise combined with an endless suppy of individual characteristics of the subject who performs the exercise. The authors of the original study also point this out and suggest that a model based approach may be more practical i.e. recording whether a subjects performance deviates from the correct form.
