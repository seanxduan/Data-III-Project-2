#Initial Data Formatting
#Sean Duan
library(tidyverse)
library(tree)
library(ISLR)
library(randomForest)
library(gbm)
library(MCMCpack)
library(mltools)
library(data.table)
set.seed(1)


#first read in the data
xdat<-read.table("Xtrain.txt")
ydat<-read.table("Ytrain.txt")
ydat<-rename(ydat, class = V1)
ydat$class<-as.factor(ydat$class)
d<-cbind(xdat,ydat)
xtest<-read.table("Xtest.txt")

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

tmp <- predict(d_bag, d, "prob")
tmz2<-tmp+.00000001
tmz2
Yhot <- one_hot(as.data.table(as.factor(d$class)))
Yhot

set.seed(1)
#ppred here is same as our tmp matrice
CE = -sum(colSums(Yhot*log(tmz2)))
CE

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

moderror
bestmodel
#took a while to run!

#best ntrees is entry 1, 300
#test on our test data
d_rf=randomForest(class~.,data=d[train,] ,mtry=4, ntree=300, importance =TRUE, probability = T)
yhat.bag<-predict(d_rf , newdata=d[-train,])


1-mean(yhat.bag==class_test)
#5.26% class err
importance(d_rf)
varImpPlot(d_rf)
#For gini
#v1, v5, v2, v3, v4, and v9

#for acc
#v1, v9, v2, v3, v4

## lets try the onehot encoding 
yhat.bag2<-predict(d_rf , newdata=d[-train,], probability = T)
attr(yhat.bag2, "probabilities")

#try this again, from xtackxchange
tmp2 <- predict(d_rf, xtest, "prob")
#above is the matrix for our test data
tmp <- predict(d_rf, d, "prob")

class(tmp)
tmz<-as.matrix(tmp)
class(tmz)
tmz

##impute some values(can't have pure 0's in our matrice)
tmz2<-tmz+.00000001
tmz2

#test using code from wikle

Yhot <- one_hot(as.data.table(as.factor(d$class)))
Yhot

set.seed(1)
#ppred here is same as our tmp matrice
CE = -sum(colSums(Yhot*log(tmz2)))
CE

