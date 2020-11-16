# STAT 8330 project #2
# David Reynolds
# Multinomial logistic regression

library(dplyr)
library(caret)

# Read in the data
setwd("~/Downloads/Documents/GitHub/Data-III-Project-2/Data")
x <- read.table("Xtrain.txt")
y <- read.table("Ytrain.txt")

y <- rename(y, class = V1)
y$class <- as.factor(y$class)
dat <- cbind(x, y)

# Fit a multinomial logistic regression model
multinom_control <- trainControl(method = "repeatedcv", number = 5, repeats = 10)

set.seed(1)
multinom <- train(class ~., data = dat, method = "multinom", trControl = multinom_control, trace = FALSE)
multinom

# Report the confusion matrix
confusionMatrix(multinom)
