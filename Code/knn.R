# STAT 8330 project #2
# David Reynolds
# KNN

library(caret)
library(class)
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

## KNN

# Implement 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the classifier
set.seed(1)
knn <- caret::train(author ~., data = train, method = "knn", trControl = train_control, 
             preProcess = c("center", "scale"), tuneLength = 20)
knn

# Plot of number of neighbors vs. accuracy
plot(knn)

# Report the confusion matrix and accuracy (0.7316)
confusionMatrix(knn)

# Calculate the training cross-entropy (3757.638)
knn_pred <- predict(knn, newdata = train, type = "prob")
knn_pred <- as.matrix(knn_pred)
knn_ce <- -sum(colSums(y_hot*log(knn_pred + 1e-15)))

# Try smaller values of k
set.seed(2)
knn_errors = rep(NA, 4)
for (i in 1:4) {
  knn = caret::train(author ~., data = train, method = "knn", trControl = train_control, 
              preProcess = c("center", "scale"), tuneGrid = data.frame(k = i))
  knn_errors[i] = as.numeric(knn$results[2])
}

# Train the classifier using k = 2
set.seed(3)
knn <- caret::train(author ~., data = train, method = "knn", trControl = train_control, 
                    preProcess = c("center", "scale"), tuneGrid = data.frame(k = 2))

# Report the confusion matrix and accuracy (0.727)
confusionMatrix(knn)

# Calculate the training cross-entropy (1590.08)
knn_pred <- predict(knn, newdata = train, type = "prob")
knn_pred <- as.matrix(knn_pred)
knn_ce <- -sum(colSums(y_hot*log(knn_pred + 1e-15)))

# Save the test prediction probabilities
Ppred4 <- predict(knn, newdata = test, type = "prob")
Ppred4 <- as.matrix(Ppred4)
save(Ppred4, file = "Ppred4.RData")
