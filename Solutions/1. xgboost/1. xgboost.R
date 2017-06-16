#-------------------------------------------------------------------------------
# Consumer Loan Default Prediction
# Data Source    : GitHub h2o
# Problem Type   : Classification
# Training Obs   : 114,908
# Validation Obs :  24,498
# Test Obs       :  24,581
# Technique      : xgboost
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Environment setup
#-------------------------------------------------------------------------------
# Clear console
rm(list=ls(all=TRUE))
gc()

# Set working directory
setwd("E:/Vikas_Agrawal/POC/Fraud")
out_dir <- "Solutions/1. xgboost"

# Load libraries
pkgs <- c("data.table", "caret", "xgboost")
sapply(pkgs, require, character.only=TRUE)

# Record start time
startTime <- Sys.time()

#-------------------------------------------------------------------------------
# Data Processing
#-------------------------------------------------------------------------------
# Import dataset
rawData <- fread("https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv")

# Lex Coding
for(f in names(rawData)) {
  tmpClass <- class(rawData[[f]])
  if(tmpClass == "character" | tmpClass == "factor") {
    tmpVals <- as.numeric(factor(rawData[[f]]))
    rawData[[f]] <- (tmpVals - min(tmpVals)) / (max(tmpVals) - min(tmpVals))
  }
}

# Id variables
outcome_name <- "bad_loan"
feature_names <- setdiff(names(rawData), outcome_name)

# Split raw dataset into training and other datasets
set.seed(1718)
trainIndex <- sample(nrow(rawData), 114908)
train_dt <- rawData[ trainIndex, ]
other_dt <- rawData[-trainIndex, ]

# Split other dataset into validation and test datasets
set.seed(1718)
valIndex <- sample(nrow(other_dt), 24498)
val_dt <- other_dt[ valIndex, ]
test_dt <- other_dt[-valIndex, ]

# Check proportion of the outcome
prop.table(table(rawData[[outcome_name]]))
prop.table(table(train_dt[[outcome_name]]))
prop.table(table(val_dt[[outcome_name]]))
prop.table(table(test_dt[[outcome_name]]))

# Convert data.table into data.frame
train_df <- data.frame(train_dt)
val_df <- data.frame(val_dt)
test_df <- data.frame(test_dt)

# Remove unwanted data from the session
rm(list=setdiff(ls(), c("out_dir", "startTime", "outcome_name", "feature_names", "train_df", "val_df", "test_df")))
gc()

#-------------------------------------------------------------------------------
# Model Development
#-------------------------------------------------------------------------------
a <- Sys.time()

# Set tuning parameters
param0 <- list(booster = "gbtree"
  , eta = 0.03
  , max_depth = 3
  , min_child_weight = 10
  , subsample = 0.8
  , colsample_bytree = 0.75
  , objective = "binary:logistic"
  , eval_metric = "auc"
)

# Data
dtrain <- xgb.DMatrix(data=data.matrix(train_df[, feature_names]), label=train_df[, outcome_name], missing=NA)
dval <- xgb.DMatrix(data=data.matrix(val_df[, feature_names]), label=val_df[, outcome_name], missing=NA)
dtest <- xgb.DMatrix(data=data.matrix(test_df[, feature_names]), label=test_df[, outcome_name], missing=NA)
watchlist <- list(val=dval, test=dtest)

# Fit CV model
set.seed(1718)
tmpPerf <- capture.output(tmpFit <- xgb.train(params=param0, data=dtrain, nrounds=6000, watchlist=watchlist, maximize=TRUE))
tmpPerf <- gsub("\\tval-auc", "", tmpPerf)
tmpPerf <- gsub("\\ttest-auc", "", tmpPerf)
tmpPerf <- gsub("[[]", "", tmpPerf)
tmpPerf <- gsub("]", "", tmpPerf)
tmpPerf <- lapply(tmpPerf, function(x) data.table(t(as.numeric(unlist(strsplit(x, ":"))))))
tmpPerf <- rbindlist(tmpPerf)
setnames(tmpPerf, c("nrounds", "val_auc", "test_auc"))
tmpPerf[, nrounds:=nrounds+1]
setorder(tmpPerf, -val_auc)
tmpPerf
b <- Sys.time()
difftime(b, a)

# Save session
save.image(paste0(out_dir, "/Session.RData"))
