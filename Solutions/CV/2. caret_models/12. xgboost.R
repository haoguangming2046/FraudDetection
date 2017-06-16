#-------------------------------------------------------------------------------
# Allstate Claims Severity
# Data Source   : Kaggle Competition
# Problem Type  : Regression
# Training Obs  : 188,318
# Testing Obs   : 125,546
#   Public Obs  :  37,664
#   Private Obs :  87,882
# nFolds        : (188318/37664)~5 folds might be ideal for cross-validation
# Technique     : xgboost
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Environment setup
#-------------------------------------------------------------------------------
# Clear console
rm(list=ls(all=TRUE))
gc()

# Set working directory
setwd("/home/rstudio/Dropbox/Public/Allstate_Claims_Severity")
# setwd("E:/Education/Kaggle/Allstate_Claims_Severity")
out_dir <- "Solutions/12. xgboost"

# Load libraries
# library(RStudioAMI)
# linkDropbox()
pkgs <- c("data.table", "caret", "xgboost")
# install.packages(pkgs)
sapply(pkgs, require, character.only=TRUE)

# Record start time
startTime <- Sys.time()


#-------------------------------------------------------------------------------
# Data processing
#-------------------------------------------------------------------------------
# Load and combine datasets
train_dt <- fread("Inputs/train.csv")
test_dt <- fread("Inputs/test.csv")
all_dt <- rbind(train_dt, test_dt, fill=TRUE)

# Lex Coding
for(f in names(all_dt)) {
  tmpClass <- class(all_dt[[f]])
  if(tmpClass == "character" | tmpClass == "factor") {
    tmpVals <- as.numeric(factor(all_dt[[f]]))
    all_dt[[f]] <- (tmpVals - min(tmpVals)) / (max(tmpVals) - min(tmpVals))
  }
}

# Id variables
id_var <- "id"
outcome_var <- "loss"
feature_var <- setdiff(names(all_dt), c(id_var, outcome_var))

# Modify outcome to normalize the curve
all_dt[[outcome_var]] <- log(all_dt[[outcome_var]] + 39)

# Split data into train and test
train_df <- data.frame(all_dt[!is.na(all_dt[[outcome_var]]), ])
test_df <- data.frame(all_dt[is.na(all_dt[[outcome_var]]), ])

# Create cross-validation folds
set.seed(1718)
train_df[, "cv_index"] <- createFolds(train_df[, outcome_var], k=5, list=FALSE)

# Clear console
rm(list=setdiff(ls(), c("startTime", "out_dir", "id_var", "outcome_var", "feature_var", "train_df", "test_df")))
gc()


#-------------------------------------------------------------------------------
# xgboost
#-------------------------------------------------------------------------------
# Evaluate mean absolute error in xgboost
evalerror <- function(preds, dtrain) {
  labels <- exp(getinfo(dtrain, "label")) - 39
  preds <- exp(preds) - 39
  if(sum(preds < 0) > 0) preds[preds < 0] <- 0
  err <- round(mean(abs(preds - labels)), 2)
  return(list(metric = "mae", value = err))
}

# Set tuning parameters
param0 <- list(booster = "gbtree"
  , eta = 0.0005
  , max_depth = 8
  , min_child_weight = 110
  , subsample = 0.8
  , colsample_bytree = 1
  , nrounds = 50000
  , objective = "reg:linear"
  , eval_metric = evalerror
)

# Find best nrounds using cross-validation
cv_perf_list <- list()
cv_model_list <- list()
for(j in 1:5) {
  print(j)
  a <- Sys.time() 
  # CV index
  train_index <- train_df$cv_index != j
  val_index <- train_df$cv_index == j

  # Data
  dtrain <- xgb.DMatrix(data=data.matrix(train_df[train_index, feature_var]), label=train_df[train_index, outcome_var])
  dval <- xgb.DMatrix(data=data.matrix(train_df[val_index, feature_var]), label=train_df[val_index, outcome_var])
  watchlist <- list(val=dval)

  # Fit CV model
  set.seed(1718)
  tmpPerf <- capture.output(tmpFit <- xgb.train(params=param0, data=dtrain, nrounds=param0$nrounds, watchlist=watchlist, maximize=FALSE))
  tmpPerf <- gsub(paste0("\\tval-", "mae"), "", tmpPerf)
  tmpPerf <- gsub("[[]", "", tmpPerf)
  tmpPerf <- gsub("]", "", tmpPerf)
  tmpPerf <- lapply(tmpPerf, function(x) data.table(t(as.numeric(unlist(strsplit(x, ":"))))))
  tmpPerf <- rbindlist(tmpPerf)
  setnames(tmpPerf, c("nrounds", paste0("val_", "mae")))
  tmpPerf[, nrounds:=nrounds+1]
  tmpPerf[, cv_index:=j]
  cv_perf_list[[j]] <- tmpPerf
  cv_model_list[[j]] <- tmpFit
  b <- Sys.time()
  print(tmpPerf[which.min(val_mae), ])
  print(difftime(b, a))
}

cv_perf <- rbindlist(cv_perf_list)
cv_error <- cv_perf[, .(mae=mean(val_mae)), by="nrounds"]
setorder(cv_error, mae)
best_nrounds <- cv_error[1, nrounds]

# Predict folds using best_nrounds
for(j in 1:5) {
  print(j)
  # Predict
  val_index <- train_df$cv_index == j
  dval <- xgb.DMatrix(data=data.matrix(train_df[val_index, feature_var]), label=train_df[val_index, outcome_var])
  train_df$pred[val_index] <- predict(object=cv_model_list[[j]], newdata=dval, ntreelimit=best_nrounds)
}

# Prepare output
cv_pred <- train_df[, c(id_var, "cv_index", outcome_var, "pred")]
cv_pred <- data.table(cv_pred)
cv_pred[, loss:=exp(loss)-39]
cv_pred[, pred:=exp(pred)-39]
cv_pred[pred < 0, pred:=0]

#-------------------------------------------------------------------------------
# Fit final model
#-------------------------------------------------------------------------------
# Data
dtrain <- xgb.DMatrix(data=data.matrix(train_df[, feature_var]), label=train_df[, outcome_var])
dtest <- xgb.DMatrix(data=data.matrix(test_df[, feature_var]))
watchlist <- list(train=dtrain)

# Fit final model
set.seed(1718)
finalFit <- xgb.train(params=param0, data=dtrain, nrounds=param0$nrounds, maximize=FALSE, watchlist=watchlist)

# Predict test datasets
testPred <- predict(object=finalFit, newdata=dtest, ntreelimit=best_nrounds)
testPred <- exp(testPred) - 39

# Store output
# submission <- fread("Inputs/sample_submission.csv")
submission <- data.table(id=test_df[[id_var]], loss=testPred)
write.csv(submission, paste0(out_dir, "/submission.csv"), row.names=FALSE)

# Record end time
endTime <- Sys.time()
timeTaken <- difftime(endTime, startTime)

# Store outputs
save(cv_perf, best_nrounds, cv_pred, timeTaken, file=paste0(out_dir, "/xgboost.RData"))
