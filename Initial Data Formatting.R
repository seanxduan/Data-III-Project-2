#Initial Data Formatting
#Sean Duan
library(tidyverse)
#first read in the data
xdat<-read.table("Xtrain.txt")
ydat<-read.table("Ytrain.txt")
ydat<-rename(ydat, class = V1)
ydat$class<-as.factor(ydat$class)
d<-cbind(xdat,ydat)

#check to see if any NA's
apply(d, 2, function(x) any(is.na(x)))
#doesn't look like we have to do any more data cleaning?