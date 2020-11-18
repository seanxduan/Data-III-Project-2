# STAT 8330 project #2
# David Reynolds
# Multinomial logistic regression

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

## Multinomial logistic regression

# Implement 5-fold repeated cross-validation
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 10)

# Fit the model
set.seed(1)
multinom <- train(author ~., data = train, method = "multinom", trControl = train_control, trace = FALSE)

# Report the confusion matrix and accuracy (0.5394)
confusionMatrix(multinom)

# Calculate the cross-entropy (39484.66)
multinom_pred <- predict(multinom, newdata = test, type = "prob")
multinom_pred <- as.matrix(multinom_pred)
multinom_ce <- -sum(colSums(y_hot*log(multinom_pred + 1e-15)))
