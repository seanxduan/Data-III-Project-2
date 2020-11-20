# STAT 8330 project #2
# David Reynolds
# KNN

library(caret)
library(class)
library(data.table)
library(dplyr)
library(ggplot2)
library(mltools)

# Read in the data
setwd("~/Downloads/Documents/GitHub/Data-III-Project-2/Data")
x_train <- read.table("Xtrain.txt")
y_train <- read.table("Ytrain.txt")
y_train <- rename(y_train, author = V1)
y_train$author <- as.factor(y_train$author)
train <- cbind(x_train, y_train)
x_test <- read.table("Xtest.txt")
y_hot <- one_hot(as.data.table(train$author))

## KNN

# Train the classifier and estimate the test error using 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

knn_accuracy = rep(NA, 15)
for (i in 1:15) {
  knn = caret::train(author ~., data = train, method = "knn", trControl = train_control, 
                     preProcess = c("center", "scale"), tuneGrid = data.frame(k = i))
  knn_accuracy[i] = as.numeric(knn$results[2])
}

# Calculate the training cross-entropy for range of k
set.seed(2)
knn_ce_train = rep(NA, 15)
for (i in 1:15) {
  knn = caret::train(author ~., data = train, method = "knn", preProcess = c("center", "scale"), 
                     tuneGrid = data.frame(k = i))
  knn_pred = predict(knn, newdata = train, type = "prob")
  knn_pred = as.matrix(knn_pred)
  knn_ce_train[i] = -sum(colSums(y_hot*log(knn_pred + 1e-15)))
}

# Train the classifier and estimate the "test CE" using 5-fold cross-validation
K <- 5
I <- 15
set.seed(3)
folds <- sample(1:K, nrow(train), replace = TRUE)
knn_ce_test <- matrix(data = NA, nrow = K, ncol = I)

for (k in 1:K) {
  cv_train = train[folds != k, ]
  cv_test = train[folds == k, ]
  y_hot = one_hot(as.data.table(cv_test$author))
  for (i in 1:15) {
    knn = caret::train(author ~., data = cv_train, method = "knn", preProcess = c("center", "scale"), 
                       tuneGrid = data.frame(k = i))
    knn_pred = predict(knn, newdata = cv_test, type = "prob")
    knn_pred = as.matrix(knn_pred)
    knn_ce_test[k,i] = -sum(colSums(y_hot*log(knn_pred + 1e-15)))
  }
  knn_ce_cv <<- colMeans(knn_ce_test)
}

# Plot the number of neighbors vs. cross-validated CE
k_ce <- data.frame(ce = knn_ce_cv)
k_ce$k <- rownames(k_ce)
k_ce$k <- as.numeric(k_ce$k)

ggplot(data = k_ce, aes(x = k, y = ce, group = 1)) + geom_line() + xlab("K") + ylab("Cross-validated CE") + 
  ggtitle("Number of neighbors vs. cross-validated CE") + theme_minimal()

# Train the best classifier on the full training set
knn <- caret::train(author ~., data = train, method = "knn", preProcess = c("center", "scale"),
                    tuneGrid = data.frame(k = 15))

# Report the confusion matrix and accuracy (0.6704)
confusionMatrix(knn)

# Calculate the training cross-entropy (6223.914)
y_hot <- one_hot(as.data.table(train$author))
knn_pred <- predict(knn, newdata = train, type = "prob")
knn_pred <- as.matrix(knn_pred)
knn_ce <- -sum(colSums(y_hot*log(knn_pred + 1e-15)))

# Save the test prediction probabilities
Ppred4 <- predict(knn, newdata = x_test, type = "prob")
Ppred4 <- as.matrix(Ppred4)
save(Ppred4, file = "Ppred4.RData")
