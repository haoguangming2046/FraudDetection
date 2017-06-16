#-------------------------------------------------------------------------------
# Consumer Loan Default Prediction
# Data Source    : GitHub h2o
# Problem Type   : Classification
# Total Obs      : 163,987
# Training Obs   : 114,791 (70%)
# Test Obs       :  49,196 (30%)
# Technique      : h2o with anomaly detection
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Environment setup
#-------------------------------------------------------------------------------
# Clear console
rm(list=ls(all=TRUE))
gc()

# Set working directory
setwd("E:/Education/POC/Fraud")
out_dir <- "Solutions/h2o_anomaly"

# Load libraries
pkgs <- c("data.table", "caret", "R.utils", "doSNOW", "pROC", "h2o")
sapply(pkgs, require, character.only=TRUE)

# Record start time
startTime <- Sys.time()

# Set number of cross-validation folds
nFolds <- 5

#-------------------------------------------------------------------------------
# Data Processing
#-------------------------------------------------------------------------------
# Import dataset
# rawData <- fread("https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv")
# save(rawData, file="Input/rawData.RData")
load("Input/rawData.RData")

# Lex Coding
for(f in names(rawData)) {
  tmpClass <- class(rawData[[f]])
  if(tmpClass == "character" | tmpClass == "factor") {
    tmpVals <- as.numeric(factor(rawData[[f]]))
    rawData[[f]] <- (tmpVals - min(tmpVals)) / (max(tmpVals) - min(tmpVals))
  }
}

# Missing value treatment
for(f in names(rawData)) {
  if(any(is.na(rawData[[f]]))) {
    tmpVals <- median(rawData[[f]], na.rm=TRUE)
    rawData[[f]][is.na(rawData[[f]])] <- tmpVals
  }
}

# Convert outcome variable to factor
# rawData[, bad_loan:=ifelse(bad_loan==0, "No", "Yes")]
# rawData[, bad_loan:=factor(bad_loan)]

# Id variables
outcome_name <- "bad_loan"
feature_names <- setdiff(names(rawData), outcome_name)

# Set column order
setcolorder(rawData, c(feature_names, outcome_name))

# Split raw dataset into training and other datasets
set.seed(1718)
trainIndex <- sample(nrow(rawData), 114791)
train_dt <- rawData[ trainIndex, ]
test_dt  <- rawData[-trainIndex, ]

# Check proportion of the outcome
prop.table(table(rawData[[outcome_name]]))
prop.table(table(train_dt[[outcome_name]]))
prop.table(table(test_dt[[outcome_name]]))

# Convert data.table into data.frame
train_df <- data.frame(train_dt)
test_df <- data.frame(test_dt)

# Create cross-validation folds
set.seed(1718)
train_df[, "cv_index"] <- createFolds(train_df[, outcome_name], k=nFolds, list=FALSE)

# Remove unwanted data from the session
rm(list=setdiff(ls(), c("out_dir", "startTime", "outcome_name", "feature_names", "train_df", "test_df", "nFolds")))
gc()

#-------------------------------------------------------------------------------
# Get benchmark score
#-------------------------------------------------------------------------------
library(randomForest)

set.seed(1718)
rf_model <- randomForest(x=train_df[, feature_names]
  , y=as.factor(train_df[, outcome_name])
  , importance=TRUE
  , ntree=20
  , mtry = 3
  )

test_predictions <- predict(rf_model, newdata=test_df[, feature_names], type="prob")

library(pROC)
auc_rf <- roc(response=test_df[, outcome_name], predictor=test_predictions[, 2])
plot(auc_rf, print.thres="best", main=paste('AUC:', round(auc_rf$auc[[1]], 3)))
abline(h=1, col='blue')
abline(h=0, col='green')

#-------------------------------------------------------------------------------
# Build autoencoder model
#-------------------------------------------------------------------------------
library(h2o)
localH2O <- h2o.init()
train.hex <- as.h2o(train_df, destination_frame="train.hex")

set.seed(1718)
train.dl <- h2o.deeplearning(x = feature_names
  , training_frame = train.hex
  , autoencoder = TRUE
  , reproducible = T
  , seed = 1718
  , hidden = c(13, 12, 13)
  , epochs = 50)

train.anon <- h2o.anomaly(train.dl, train.hex, per_feature=FALSE)
head(train.anon)
err <- as.data.frame(train.anon)
plot(sort(err$Reconstruction.MSE))
tail(sort(err$Reconstruction.MSE), 100)
plot(sort(err$Reconstruction.MSE[err$Reconstruction.MSE < 0.2]))

#-------------------------------------------------------------------------------
# Use the easy portion and model with random forest using same settings
#-------------------------------------------------------------------------------
train_df_auto <- train_df[err$Reconstruction.MSE < 0.1, ]

set.seed(1718)
rf_model <- randomForest(x=train_df_auto[, feature_names]
  , y=as.factor(train_df_auto[, outcome_name])
  , importance=TRUE
  , ntree=20
  , mtry = 3
  )

test_predictions_known <- predict(rf_model, newdata=test_df[, feature_names], type="prob")

library(pROC)
auc_rf <- roc(response=test_df[, outcome_name], predictor=test_predictions_known[, 2])
plot(auc_rf, print.thres="best", main=paste('AUC:', round(auc_rf$auc[[1]], 3)))
abline(h=1, col='blue')
abline(h=0, col='green')

#-------------------------------------------------------------------------------
# Use the easy portion and model with random forest using same settings
#-------------------------------------------------------------------------------
train_df_auto <- train_df[err$Reconstruction.MSE >= 0.1 & err$Reconstruction.MSE < 0.2, ]

set.seed(1718)
rf_model <- randomForest(x=train_df_auto[, feature_names]
  , y=as.factor(train_df_auto[, outcome_name])
  , importance=TRUE
  , ntree=20
  , mtry = 3
  )

test_predictions_unknown <- predict(rf_model, newdata=test_df[, feature_names], type="prob")

library(pROC)
auc_rf <- roc(response=test_df[, outcome_name], predictor=test_predictions_unknown[, 2])
plot(auc_rf, print.thres="best", main=paste('AUC:', round(auc_rf$auc[[1]], 3)))
abline(h=1, col='blue')
abline(h=0, col='green')

#-------------------------------------------------------------------------------
# Bag both results set and measure final AUC score
#-------------------------------------------------------------------------------
valid_all <- (test_predictions_known[, 2] + test_predictions_unknown[, 2]) / 2
auc_rf <- roc(response=test_df[, outcome_name], predictor=valid_all)
plot(auc_rf, print.thres="best", main=paste('AUC:', round(auc_rf$auc[[1]], 3)))
abline(h=1, col='blue')
abline(h=0, col='green')
