library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(ggplot2)
#install.packages("xgboost")

data <- read.csv("fraud-detection1.csv")


data <- data[, c("TransactionAmount", "IsForeignTransaction", "NumChargebacks", 
                 "HasEmailDomainBlacklisted", "Fraud")]


data$Fraud <- as.factor(data$Fraud)


ggplot(data, aes(x = TransactionAmount, fill = Fraud)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Transaction Amount Distribution by Fraud", x = "Transaction Amount", y = "Count") +
  theme_minimal()


ggplot(data, aes(x = Fraud, fill = Fraud)) +
  geom_bar() +
  labs(title = "Fraud Count", x = "Fraud (0 = No, 1 = Yes)", y = "Count") +
  theme_minimal()


set.seed(123)
trainIndex <- createDataPartition(data$Fraud, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Model 1: 
log_model <- glm(Fraud ~ ., data = trainData, family = binomial)
log_preds <- predict(log_model, testData, type = "response")
log_class <- ifelse(log_preds > 0.5, 1, 0)
log_auc <- roc(testData$Fraud, as.numeric(log_preds))$auc

# Model 2: 
rf_model <- randomForest(Fraud ~ ., data = trainData, ntree = 100)
rf_preds <- predict(rf_model, testData)
rf_auc <- roc(testData$Fraud, as.numeric(predict(rf_model, testData, type = "prob")[,2]))$auc

# Model 3: 
train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -5]), label = as.numeric(trainData$Fraud) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -5]))

xgb_model <- xgboost(data = train_matrix, nrounds = 100, objective = "binary:logistic", verbose = 0)
xgb_preds <- predict(xgb_model, test_matrix)
xgb_auc <- roc(testData$Fraud, xgb_preds)$auc

# Model 4:
svm_model <- svm(Fraud ~ ., data = trainData, probability = TRUE)
svm_preds <- predict(svm_model, testData, probability = TRUE)
svm_prob <- attr(svm_preds, "probabilities")[,2]
svm_auc <- roc(testData$Fraud, svm_prob)$auc


results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost)", "SVM"),
  AUC = c(log_auc, rf_auc, xgb_auc, svm_auc)
)

print(results)


ggplot(results, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity", width = 0.7) +
  labs(title = "Model Comparison: AUC Scores", x = "Model", y = "AUC") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")


roc_log <- roc(testData$Fraud, as.numeric(log_preds))
roc_rf <- roc(testData$Fraud, as.numeric(predict(rf_model, testData, type = "prob")[,2]))
roc_xgb <- roc(testData$Fraud, xgb_preds)
roc_svm <- roc(testData$Fraud, svm_prob)


plot(roc_log, col = "blue", main = "ROC Curves for Models", lwd = 2)
plot(roc_rf, col = "green", add = TRUE, lwd = 2)
plot(roc_xgb, col = "orange", add = TRUE, lwd = 2)
plot(roc_svm, col = "purple", add = TRUE, lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"),
       col = c("blue", "green", "orange", "purple"), lwd = 2)
