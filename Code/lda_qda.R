# STAT 8330 project #2
# David Reynolds
# LDA and QDA

library(caret)
library(data.table)
library(dplyr)
library(mltools)

# Read in the data
setwd("~/Downloads/Documents/GitHub/Data-III-Project-2/Data")
x <- read.table("Xtrain.txt")
y <- read.table("Ytrain.txt")
y <- rename(y, author = V1)
y$author <- as.factor(y$author)
train <- cbind(x, y)
test <- read.table("Xtest.txt")
y_hot <- one_hot(as.data.table(train$author))

## LDA

# Implement 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Fit the model
set.seed(1)
lda <- caret::train(author ~., data = train, method = "lda", trControl = train_control, metric = "Accuracy")

# Report the confusion matrix and accuracy (0.4988)
confusionMatrix(lda)

# Calculate the training cross-entropy (14464.28)
lda_pred <- predict(lda, newdata = train, type = "prob")
lda_pred <- as.matrix(lda_pred)
lda_ce <- -sum(colSums(y_hot*log(lda_pred + 1e-15)))

# Save the test prediction probabilities
Ppred2 <- predict(lda, newdata = test, type = "prob")
Ppred2 <- as.matrix(Ppred2)
save(Ppred2, file = "Ppred2.RData")

## QDA

# Fit the model
set.seed(2)
qda <- caret::train(author ~., data = train, method = "qda", trControl = train_control, metric = "Accuracy")

# Report the confusion matrix and accuracy (0.4088)
confusionMatrix(qda)

# Calculate the training cross-entropy (20086.22)
qda_pred <- predict(qda, newdata = train, type = "prob")
qda_pred <- as.matrix(qda_pred)
qda_ce <- -sum(colSums(y_hot*log(qda_pred + 1e-15)))

# Save the test prediction probabilities
Ppred3 <- predict(qda, newdata = test, type = "prob")
Ppred3 <- as.matrix(Ppred3)
save(Ppred3, file = "Ppred3.RData")
