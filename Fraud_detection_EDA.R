
#install.packages("ggplot2")      
#install.packages("dplyr")         
#install.packages("tidyr")         
#install.packages("corrplot")     
#install.packages("DataExplorer") 
library(ggplot2)      
library(dplyr)     
library(tidyr)       
library(corrplot)     
library(DataExplorer) 


mydata <- read.csv("fraud-detection1.csv")


str(mydata)             
summary(mydata)         
head(mydata)           


colSums(is.na(mydata))  


mydata <- mydata %>% select(-Age)


# Distribution of Transaction Amount
ggplot(mydata, aes(x = TransactionAmount)) + 
  geom_histogram(binwidth = 100, fill = "blue", alpha = 0.7) + 
  labs(title = "Distribution of Transaction Amounts", x = "Transaction Amount", y = "Frequency")

# Account Age and Number of Chargebacks
ggplot(mydata, aes(x = AccountAgeDays, y = NumChargebacks)) +
  geom_point(alpha = 0.6, color = "darkred") + 
  labs(title = "Account Age vs Chargebacks", x = "Account Age (days)", y = "Number of Chargebacks")

# Transaction count by hour of the day
ggplot(mydata, aes(x = TransactionHour)) + 
  geom_histogram(binwidth = 1, fill = "purple", alpha = 0.7) + 
  labs(title = "Transactions per Hour of the Day", x = "Hour", y = "Number of Transactions")


# Count plot for MerchantCategory
ggplot(mydata, aes(x = MerchantCategory)) +
  geom_bar(fill = "green", alpha = 0.7) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Transaction Count by Merchant Category", x = "Merchant Category", y = "Count")

# Count plot for PaymentMethod
ggplot(mydata, aes(x = PaymentMethod)) +
  geom_bar(fill = "orange", alpha = 0.7) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(title = "Transaction Count by Payment Method", x = "Payment Method", y = "Count")

# Correlation analysis for numerical variables
# Calculate correlations
numeric_vars <- mydata %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Correlation plot
corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.7, tl.col = "black")

# Detecting multicollinearity 
# Pairwise correlation plot for selected numerical variables
pairs(~ TransactionAmount + DailyTransactionCount + NumChargebacks + AccountAgeDays, data = mydata, 
      main = "Pairwise Scatter Plots")

# Checking for outliers
# Boxplot for Transaction Amount
ggplot(mydata, aes(x = "", y = TransactionAmount)) + 
  geom_boxplot(fill = "blue", alpha = 0.7) + 
  labs(title = "Boxplot of Transaction Amount", x = "", y = "Transaction Amount")

# Plot Fraud distribution
ggplot(mydata, aes(x = as.factor(Fraud))) +
  geom_bar(fill = "red", alpha = 0.7) +
  labs(title = "Fraudulent vs Legitimate Transactions", x = "Fraud (0: Legitimate, 1: Fraudulent)", y = "Count")

# Investigating categorical fraud data
# Fraud rates by payment method
ggplot(mydata, aes(x = PaymentMethod, fill = as.factor(Fraud))) +
  geom_bar(position = "fill", alpha = 0.7) +
  labs(title = "Fraud Rate by Payment Method", x = "Payment Method", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


#Next week
#  Residual plots (optional: if building a model)
# Residuals vs Fitted for potential regression models
# res <- residuals(model)
# ggplot(data = data.frame(res, Fitted = fitted(model)), aes(x = Fitted, y = res)) +
#   geom_point() +
#   labs(title = "Residuals vs Fitted", x = "Fitted Values", y = "Residuals")

# Automated EDA (optional)
#DataExplorer::create_report(mydata)

# End of EDA
#-----------------------------------------------------------
# Load necessary libraries
# List of required packages


library(tidyverse)
library(caTools)
library(ROCR)
library(ggplot2)

# Step 1: Load and Inspect Data
fraud_data <- read.csv("fraud-detection1.csv")  # Replace with your dataset path
head(fraud_data)
summary(fraud_data)

# Step 2: Data Preprocessing
# Select key columns
fraud_data <- fraud_data %>%
  select(TransactionAmount, IsForeignTransaction, NumChargebacks, HasEmailDomainBlacklisted, Fraud)

# Convert 'Fraud' column to a factor for classification
fraud_data$Fraud <- as.factor(fraud_data$Fraud)

# Split data into training (70%) and testing (30%) sets
set.seed(123)
split <- sample.split(fraud_data$Fraud, SplitRatio = 0.7)
train_data <- subset(fraud_data, split == TRUE)
test_data <- subset(fraud_data, split == FALSE)

# Step 3: Data Visualization
# Transaction Amount Distribution
ggplot(fraud_data, aes(x = TransactionAmount, fill = Fraud)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Transaction Amount Distribution by Fraud Status", x = "Transaction Amount", y = "Count") +
  theme_minimal()

# Fraud Count Bar Plot
ggplot(fraud_data, aes(x = Fraud, fill = Fraud)) +
  geom_bar() +
  labs(title = "Fraudulent vs Non-Fraudulent Transactions", x = "Fraud Status", y = "Count") +
  theme_minimal()

# Step 4: Train Logistic Regression Model
logistic_model <- glm(Fraud ~ ., data = train_data, family = binomial)

# Step 5: Evaluate Model Performance
# Predict probabilities on test set
predictions <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to binary outcomes (threshold = 0.5)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Calculate AUC
pred_obj <- prediction(predictions, as.numeric(test_data$Fraud))
auc <- performance(pred_obj, "auc")@y.values[[1]]
print(paste("AUC for Logistic Regression:", round(auc, 3)))

# Plot ROC Curve
roc_curve <- performance(pred_obj, "tpr", "fpr")
plot(roc_curve, col = "blue", main = "ROC Curve for Logistic Regression")
abline(a = 0, b = 1, lty = 2, col = "gray")

# Save results for comparison
logistic_results <- data.frame(Model = "Logistic Regression", AUC = round(auc, 3))

# Step 6: Optional - Add Other Models
# Train Random Forest, SVM, Gradient Boosting, and compare AUC scores (if needed)
# See provided workflow for implementation details of these models

# Step 7: Model Comparison Visualization (if multiple models are used)
# Example of AUC bar chart (if data for other models is added)
auc_scores <- rbind(logistic_results)
ggplot(auc_scores, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  labs(title = "AUC Scores for Models", x = "Model", y = "AUC") +
  theme_minimal()
#---------------------------------------------------------------------
#Random forest

library(randomForest)
library(caret)

# Data Preprocessing
# Convert Fraud column to a factor
fraud_data$Fraud <- as.factor(fraud_data$Fraud)

# Select relevant features
selected_features <- fraud_data %>%
  select(TransactionAmount, IsForeignTransaction, NumChargebacks, HasEmailDomainBlacklisted, Fraud)

# Split data into training (70%) and testing (30%) sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(selected_features$Fraud, p = 0.7, list = FALSE)
train_data <- selected_features[train_index, ]
test_data <- selected_features[-train_index, ]

# Train the Random Forest model
set.seed(123)  # For reproducibility
rf_model <- randomForest(Fraud ~ ., data = train_data, ntree = 100, mtry = 2, importance = TRUE)

# Model Summary
print(rf_model)

# Variable Importance Plot
varImpPlot(rf_model, main = "Variable Importance")

# Predictions on the test dataset
rf_predictions <- predict(rf_model, test_data, type = "response")

# Evaluate Model Performance
confusion_matrix <- confusionMatrix(rf_predictions, test_data$Fraud)
print(confusion_matrix)

# Calculate AUC and plot ROC curve
rf_probabilities <- predict(rf_model, test_data, type = "prob")[, 2]
pred <- prediction(rf_probabilities, as.numeric(test_data$Fraud) - 1)
perf <- performance(pred, "tpr", "fpr")

# Plot ROC Curve
plot(perf, col = "blue", main = "Random Forest: ROC Curve")
abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal line
auc <- performance(pred, "auc")@y.values[[1]]
legend("bottomright", legend = paste("AUC =", round(auc, 3)), col = "blue", lty = 1)

# Print AUC Score
print(paste("AUC Score:", round(auc, 3)))

# Save Results
write.csv(confusion_matrix$table, "confusion_matrix_rf.csv")
write.csv(rf_model$importance, "variable_importance_rf.csv")
#---------------------------------------------------------------------------
install.packages(c("e1071", "caret", "pROC", "ggplot2"))
library(e1071)  # For SVM
library(caret)  # For model training and evaluation
library(pROC)   # For ROC curve and AUC
library(ggplot2) # For visualization
# Install and load necessary packages
#install.packages(c("e1071", "caret", "pROC", "ggplot2"))
#install.packages("e1071")
library(e1071)  # For SVM
library(caret)  # For model training and evaluation
library(pROC)   # For ROC curve and AUC
library(ggplot2) # For visualization

# Train the SVM model with probability estimation
svm_model <- svm(Fraud ~ ., data = train_data, kernel = "linear", probability = TRUE)

# Make predictions on the test data
svm_predictions <- predict(svm_model, test_data, type = "response")  # Predicted classes
svm_predictions_prob <- predict(svm_model, test_data, type = "prob")  # Predicted probabilities

# Check if probabilities were returned as expected
if (is.matrix(svm_predictions_prob)) {
  svm_predictions_prob <- svm_predictions_prob[, 2]  # For binary classification, take the second column
} else {
  # If probabilities are not returned as a matrix, manually calculate probabilities using a logistic model or use another approach
  cat("The model did not return probabilities. Using class predictions.\n")
  svm_predictions_prob <- as.numeric(svm_predictions) - 1  # Convert class labels to numeric (fraud = 1, non-fraud = 0)
}

# Generate confusion matrix to evaluate performance
conf_matrix <- confusionMatrix(svm_predictions, test_data$Fraud)
print(conf_matrix)

# Calculate ROC curve and AUC
svm_roc <- roc(test_data$Fraud, svm_predictions_prob)
svm_auc <- auc(svm_roc)

# Plot ROC curve
plot(svm_roc, col = "blue", main = "ROC Curve - Support Vector Machine")
legend("bottomright", legend = paste("AUC =", round(svm_auc, 2)), col = "blue", lwd = 2)

# Print the AUC score
cat("\nAUC of the Model: ", round(svm_auc, 2))
#----------------------------------------------------------
#Gradient Boosting
# Install and load necessary libraries
# Install necessary libraries (if not already installed)
install.packages(c("gbm", "caret", "pROC", "dplyr"))
library(gbm)    # For Gradient Boosting
library(caret)  # For model training and evaluation
library(pROC)   # For ROC curve and AUC
library(dplyr)  # For data manipulation

# Load the dataset (replace with actual file path)
data <- read.csv("fraud-detection1.csv")

# Check the structure of the dataset to identify data types
str(data)

# Convert all required columns to appropriate data types:
# If 'MerchantCategory' is not a factor or numeric, convert it to a factor
if(!is.factor(data$MerchantCategory) && !is.numeric(data$MerchantCategory)) {
  data$MerchantCategory <- as.factor(data$MerchantCategory)
}

# If 'DeviceType' is not a factor or numeric, convert it to a factor
if(!is.factor(data$DeviceType) && !is.numeric(data$DeviceType)) {
  data$DeviceType <- as.factor(data$DeviceType)
}

# If 'Browser' is not a factor or numeric, convert it to a factor
if(!is.factor(data$Browser) && !is.numeric(data$Browser)) {
  data$Browser <- as.factor(data$Browser)
}

# Ensure the 'Fraud' column is a factor for classification (target variable)
data$Fraud <- as.factor(data$Fraud)

# Split the dataset into training (70%) and testing (30%) sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Fraud, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the Gradient Boosting model
gbm_model <- gbm(Fraud ~ ., 
                 data = train_data, 
                 distribution = "bernoulli",  # Binary classification
                 n.trees = 1000,              # Number of trees
                 interaction.depth = 3,       # Depth of trees (controls overfitting)
                 shrinkage = 0.01,            # Learning rate
                 n.minobsinnode = 10,         # Minimum number of observations per node
                 cv.folds = 5,                # Cross-validation for hyperparameter tuning
                 verbose = TRUE)

# Print the model summary
summary(gbm_model)

# Make predictions on the test data
gbm_predictions_prob <- predict(gbm_model, test_data, n.trees = 1000, type = "response")

# Predict the classes (fraud or non-fraud) using a threshold of 0.5
gbm_predictions <- ifelse(gbm_predictions_prob > 0.5, 1, 0)
gbm_predictions <- factor(gbm_predictions, levels = c(0, 1), labels = c("Non-Fraud", "Fraud"))

# Generate confusion matrix to evaluate model performance
conf_matrix <- confusionMatrix(gbm_predictions, test_data$Fraud)
print(conf_matrix)

# Calculate ROC curve and AUC
gbm_roc <- roc(test_data$Fraud, gbm_predictions_prob)
gbm_auc <- auc(gbm_roc)

# Plot ROC curve
plot(gbm_roc, col = "blue", main = "ROC Curve - Gradient Boosting Machine")
legend("bottomright", legend = paste("AUC =", round(gbm_auc, 2)), col = "blue", lwd = 2)

# Print the AUC score
cat("\nAUC of the Model: ", round(gbm_auc, 2))
