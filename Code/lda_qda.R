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
lda <- train(author ~., data = train, method = "lda", trControl = train_control, metric = "Accuracy")

# Report the confusion matrix and accuracy (0.4988)
confusionMatrix(lda)

# Calculate the cross-entropy (41218.48)
lda_pred <- predict(lda, newdata = test, type = "prob")
lda_pred <- as.matrix(lda_pred)
lda_ce <- -sum(colSums(y_hot*log(lda_pred + 1e-15)))

## QDA

# Fit the model
set.seed(2)
qda <- train(author ~., data = train, method = "qda", trControl = train_control, metric = "Accuracy")

# Report the confusion matrix and accuracy (0.4088)
confusionMatrix(qda)

# Calculate the cross-entropy (80852.37)
qda_pred <- predict(qda, newdata = test, type = "prob")
qda_pred <- as.matrix(qda_pred)
qda_ce <- -sum(colSums(y_hot*log(qda_pred + 1e-15)))
