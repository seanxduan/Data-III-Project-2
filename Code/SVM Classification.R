
library(MCMCpack)
library(mltools)
library(data.table)
library(e1071)

#Provided Code
Xtrain = read.table("c:/Users/JoeCo/Downloads/Xtrain.txt")
Ytrain = read.table("c:/Users/JoeCo/Downloads/Ytrain.txt", quote="\"", comment.char="")
training = cbind(Ytrain, Xtrain)
names(training)[1] = "Author"
training$Author = as.factor(training$Author)


Yhot = one_hot(as.data.table(as.factor(Ytrain$V1)))
Xtest = read.table("c:/Users/JoeCo/Downloads/Xtest.txt")

#Test it
set.seed(1)
Ppred <- rdirichlet(10000, c(1,1,1,1,1,1,1,1,1) )
CE = -sum(colSums(Yhot*log(Ppred)))
CE



#Some work
set.seed(1)

tune.out1 = tune(svm, Author ~., data = training, kernel = "linear",
                 ranges = list(cost = c(0.01, 0.1, 1, 10, 100)))
tune.out2 = tune(svm, Author ~., data = training, kernel = "radial", 
                 ranges = list(cost = c(.01, 1, 10, 100),
                               gamma = c(.01, 1, 10, 100)))
tune.out3 = tune(svm, Author ~., data = training, kernel = "polynomial",
                 ranges = list(cost = c(.01, 1, 10, 100),
                               degree = c(2, 3, 4)))
tune.out4 = tune(svm, Author ~., data = training, kernel = "sigmoid", 
                 ranges = list(cost = c(0.1, 1, 10, 100),
                               gamma = c(0.1, 1, 10, 100)))



summary(tune.out1)
best.mod1 = svm(Author ~., data = training, kernel = "linear",
                cost = 100, probability = T)
ypred1 = predict(best.mod1, Xtrain, probability = T)
probstrain1 = attr(ypred1, "probabilities")

CEtrain1 = -sum(colSums(Yhot*log(probstrain1)))

summary(tune.out2)
best.mod2 = svm(Author ~ ., data = training, kernel = "radial",
                cost = 10, gamma = 1, probability = T)

ypred2 = predict(best.mod2, Xtrain, probability = T)
probstrain2 = attr(ypred2, "probabilities")
CEtrain2 = -sum(colSums(Yhot*log(probstrain2)))

summary(tune.out3)
best.mod3 = svm(Author ~ ., data = training, kernel = "polynomial",
                cost = 100, degree = 3, probability = T)
ypred3 = predict(best.mod3, Xtrain, probability = T)
probstrain3 = attr(ypred3, "probabilities")
CEtrain3 = -sum(colSums(Yhot*log(probstrain3)))


summary(tune.out4)
best.mod4 = svm(Author ~., data = training, kernel = "sigmoid",
                cost = 0.1, gamma = 0.1, probability = T)
ypred4 = predict(best.mod4, Xtrain, probability = T)
probstrain4 = attr(ypred4, "probabilities")
CEtrain4 = -sum(colSums(Yhot*log(probstrain4)))

random = svm(Author ~., data = training, kernel = "sigmoid",
             cost = 100, gamma = 100, probability = T)
randompred = predict(random, Xtrain, probability = T)
randomprob = attr(randompred, "probabilities")
CErandom = -sum(colSums(Yhot*log(randomprob)))
