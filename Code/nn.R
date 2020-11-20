# STAT 8330 project #2
# David Reynolds
# Neural network

library(ANN2)
library(caret)
library(data.table)
library(dplyr)
library(mltools)

# Read in the data
setwd("~/Downloads/Documents/GitHub/Data-III-Project-2/Data")
x_train <- read.table("Xtrain.txt")
y_train <- read.table("Ytrain.txt")
y_train <- rename(y_train, author = V1)
y_train$author <- as.factor(y_train$author)
x_test <- read.table("Xtest.txt")

# Scale the predictors
rescale <- function(x, x_min, x_max) {
  r = (x - x_min)/(x_max - x_min) 
  r[is.nan(r)] = 0
  return(r)
}

x_min <- apply(x_train, 2, min)
x_max <- apply(x_train, 2, max)
x_train_scaled <- t(apply(x_train, 1, rescale, x_min, x_max))
x_test_scaled <- t(apply(x_test, 1, rescale, x_min, x_max))
train <- cbind(x_train_scaled, y_train)

# Train the neural network and estimate the "test CE" using 5-fold cross-validation (1996.641)
K <- 5
set.seed(1)
folds <- sample(1:K, nrow(train), replace = TRUE)
nn_ce <- rep(NA, K)

for (k in 1:K) {
  cv_train = train[folds != k, ]
  cv_test = train[folds == k, ]
  x_cv_train = cv_train[, 1:10]
  y_cv_train = cv_train[, 11]
  x_cv_test = cv_test[, 1:10]
  y_cv_test = cv_test[, 11]
  y_hot = one_hot(as.data.table(y_cv_test))
  set.seed(2)
  nn = neuralnetwork(X = x_cv_train, y = y_cv_train, hidden.layers = c(9, 8, 7), optim.type = "adam",
                     n.epochs = 10, activ.functions = "tanh", learn.rates = 0.05, 
                     val.prop = 0.05, verbose = FALSE)
  nn_pred = predict(nn, newdata = x_cv_test, probability = TRUE)
  nn_probs = as.data.frame(nn_pred$probabilities)
  nn_probs = nn_probs %>% 
    select("1" = class_1, "4" = class_4, "5" = class_5, "6" = class_6, "7" = class_7, "8" = class_8,
           "9" = class_9, "11" = class_11, "12" = class_12)
  nn_probs <- as.matrix(nn_probs)
  nn_ce[k] <- -sum(colSums(y_hot*log(nn_probs + 1e-15)))
}
mean(nn_ce)

# Train the best neural network on the full training set
set.seed(3)
nn <- neuralnetwork(X = x_train_scaled, y = y_train, hidden.layers = c(9, 8, 7), optim.type = "adam", 
                    n.epochs = 10, activ.functions = "tanh", learn.rates = 0.05, val.prop = 0.05, 
                    verbose = FALSE)

# Plot the loss during training
plot(nn)

# Report the confusion matrix and accuracy (0.2272)
nn_pred <- predict(nn, newdata = x_train_scaled)
nn_table <- table(truth = y_train$author, fitted = nn_pred$predictions)
sum(diag(nn_table))/length(y_train$author)

# Calculate the training cross-entropy (10049.33)
nn_pred <- predict(nn, newdata = x_train_scaled, probability = TRUE)
nn_probs <- as.data.frame(nn_pred$probabilities)

nn_probs <- nn_probs %>% 
  select("1" = class_1, "4" = class_4, "5" = class_5, "6" = class_6, "7" = class_7, "8" = class_8,
         "9" = class_9, "11" = class_11, "12" = class_12)

nn_probs <- as.matrix(nn_probs)
y_hot = one_hot(as.data.table(y_train$author))
nn_ce <- -sum(colSums(y_hot*log(nn_probs + 1e-15)))

# Save the test prediction probabilities
Ppred5 <- predict(nn, newdata = x_test_scaled, probability = TRUE)
Ppred5 <- as.data.frame(Ppred5$probabilities)

Ppred5 <- Ppred5 %>% 
  select("1" = class_1, "4" = class_4, "5" = class_5, "6" = class_6, "7" = class_7, "8" = class_8,
         "9" = class_9, "11" = class_11, "12" = class_12)

Ppred5 <- as.matrix(Ppred5)
save(Ppred5, file = "Ppred5.RData")
