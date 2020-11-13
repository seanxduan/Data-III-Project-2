#Initial Data Formatting
#Sean Duan
library(tidyverse)
library(tree)
library(ISLR)
library(randomForest)
library(gbm)
set.seed(1)


#first read in the data
xdat<-read.table("Xtrain.txt")
ydat<-read.table("Ytrain.txt")
ydat<-rename(ydat, class = V1)
ydat$class<-as.factor(ydat$class)
d<-cbind(xdat,ydat)

#check to see if any NA's
apply(d, 2, function(x) any(is.na(x)))
#doesn't look like we have to do any more data cleaning?

#prep data for split into test/train
set.seed(1)
train=sample(1:nrow(d),5000)
d_test=d[-train,]
class_test=d_test[,11]

#bagging
d_bag=randomForest(class~.,data=d , subset=train,mtry=10,importance =TRUE)
d_bag
#error rate 5.64%
#checking test MSE
yhat.bag = predict (d_bag , newdata=d[-train,])
plot(yhat.bag , class_test)
#importance of variables
importance(d_bag)
varImpPlot(d_bag)
#since gini is interpretationable, 
#we see gini best is v5, v1, v2, and v3

#acc decrease is what we're looking for b/c score improvement
#we see acc best is v1, v9, v2 and v5
table(yhat.bag,class_test)
1-mean(yhat.bag==class_test)
#6.02% class error!

#boosting
#unsure how to set distrib. lets put a pin in it until we talk to wikle/other group members.

#random forest
#we start w/ a loop to find the best m value!
d_train<-d[train,]
K=5
folds = sample(1:K,nrow(d_train),replace=T)
bestmodel<-list(NA)
moderror<-list(NA)
test_mse<-list(NA)
mvec<-seq(from=1, to=10)
yhat.bag<-list(NA)
d_rf<-list(NA)
class_err<-list(NA)

for(k in 1:K){
  CV.train = d_train[folds != k,]
  CV.test = d_train[folds == k,]
  CV.ts_y = CV.test$class
  for(i in 1:length(mvec)){
    d_rf[[i]]=randomForest(class~.,data=CV.train ,mtry=mvec[i],importance =TRUE)
    yhat.bag[[i]]<-predict(d_rf[[i]] , newdata=CV.test)
    class_err[[i]]<-(1-mean(yhat.bag[[i]]==CV.ts_y))
  }
  moderror[[k]]<-class_err[[which.min(class_err)]]
  bestmodel[[k]]<-which.min(class_err)}
moderror
bestmodel
#mvec entry #4, which is 4 is our lowest class error

#lets try ntree
K=5
bestmodel<-list(NA)
moderror<-list(NA)
test_mse<-list(NA)
yhat.bag<-list(NA)
d_rf<-list(NA)
class_err<-list(NA)
testn<-seq(from=300, to=3000, length.out = 10)
for(k in 1:K){
  CV.train = d_train[folds != k,]
  CV.test = d_train[folds == k,]
  CV.ts_y = CV.test$class
  for(i in 1:length(testn)){
    d_rf[[i]]=randomForest(class~.,data=CV.train ,mtry=4, ntree=testn[[i]], importance =TRUE)
    yhat.bag[[i]]<-predict(d_rf[[i]] , newdata=CV.test)
    class_err[[i]]<-(1-mean(yhat.bag[[i]]==CV.ts_y))
  }
  moderror[[k]]<-class_err[[which.min(class_err)]]
  bestmodel[[k]]<-which.min(class_err)}


#best ntrees is 300 across all folds
#test on our test data
fish_rf=randomForest(LSH7class~.,data=fish[training.set,] ,mtry=2, ntree=300, importance =TRUE)
yhat.bag<-predict(fish_rf , newdata=fish[-training.set,])
1-mean(yhat.bag==LSH7class.test)
#2.41% class err
importance(fish_rf)
varImpPlot(fish_rf)
#secchi lake mean, area hectacres, mean gdd, max depth most important