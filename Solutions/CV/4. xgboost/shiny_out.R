#-------------------------------------------------------------------------------
# Consumer Loan Default Prediction
# Data Source    : GitHub h2o
# Problem Type   : Classification
# Total Obs      : 163,987
# Training Obs   : 114,791 (70%)
# Test Obs       :  49,196 (30%)
# Technique      : xgboost
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Environment setup
#-------------------------------------------------------------------------------
# Clear console
rm(list=ls(all=TRUE))
gc()

# Set working directory
setwd("E:/Education/POC/Fraud")
out_dir <- "Solutions/CV/4. xgboost"

# Load libraries
pkgs <- c("data.table", "caret", "R.utils", "doSNOW", "xgboost", "pROC", "Ckmeans.1d.dp")
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

rawData[, id:=1:.N]
rawData[, term:=as.numeric(gsub(" months", "", term))]
rawData[home_ownership %in% c("ANY", "NONE", "OTHER"), home_ownership:="OTHER"]
rawData[verification_status %in% "not verified", verification_status:="0"]
rawData[verification_status %in% "verified", verification_status:="1"]
rawData[, verification_status:=as.numeric(verification_status)]

# Convert character to factor
for(f in names(rawData)) {
  if(class(rawData[[f]]) == "character") {
    rawData[[f]] <- factor(rawData[[f]])
  }
}

# Id variables
outcome_name <- "bad_loan"
feature_names <- setdiff(names(rawData), c(outcome_name, "id"))

# Set column order
setcolorder(rawData, c("id", feature_names, outcome_name))

# Split raw dataset into training and other datasets
set.seed(1718)
trainIndex <- sample(nrow(rawData), 114791)
train_dt <- rawData[ trainIndex, ]
test_dt  <- rawData[-trainIndex, ]
# setorder(train_dt, id)
# setorder(test_dt, id)

# Convert data.table into data.frame
# train_df <- data.frame(train_dt)
# test_df <- data.frame(test_dt)
# rawData <- data.frame(rawData)

save(train_dt, test_dt, rawData, file="E:/Education/Shiny/App-1/www/Data.RData")
ROC <- auc_rf
save(finalFit, feature_names, ROC, importance_matrix, modelPred, opt_cutoff, test_acc, test_AUC, best_nrounds, confMat, file="E:/Education/Shiny/App-1/data/Output.RData")
