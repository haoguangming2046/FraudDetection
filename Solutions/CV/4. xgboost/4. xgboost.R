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

# Dummy Data
dummyModel <- dummyVars(~., rawData, sep="_")
rawData <- predict(dummyModel, rawData)
rawData <- data.table(rawData)

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

# Function to plot ROC
ggroc <- function(roc, showAUC = TRUE, interval = 0.2, breaks = seq(0, 1, interval), acc=NULL) {
  require(pROC)
  if(class(roc) != "roc")
    simpleError("Please provide roc object from pROC package")
  plotx <- rev(roc$specificities)
  ploty <- rev(roc$sensitivities)

  ggplot(NULL, aes(x = plotx, y = ploty)) +
    geom_segment(aes(x = 0, y = 1, xend = 1,yend = 0), colour="red", alpha = 0.5, size=1) +
    geom_step(size=1, colour="blue") +
    scale_x_reverse(name = "Specificity",limits = c(1,0), breaks = breaks, expand = c(0.001,0.001)) +
    scale_y_continuous(name = "Sensitivity", limits = c(0,1), breaks = breaks, expand = c(0.001, 0.001)) +
    theme(axis.ticks = element_line(color = "grey80")) +
    coord_equal() +
    ggtitle("ROC") +
    annotate("text", x = interval/2, y = interval/2, vjust = 0, label = paste("AUC:", sprintf("%.3f",roc$auc), "\n", "Accuracy:", acc))
}

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
  dtrain <- xgb.DMatrix(data=data.matrix(train_df[train_index, feature_names]), label=train_df[train_index, outcome_name], missing=NA)
  dval <- xgb.DMatrix(data=data.matrix(train_df[val_index, feature_names]), label=train_df[val_index, outcome_name], missing=NA)
  watchlist <- list(val=dval)

  # Fit CV model
  set.seed(1718)
  tmpPerf <- capture.output(tmpFit <- xgb.train(params=param0, data=dtrain, nrounds=2500, watchlist=watchlist, maximize=TRUE))
  tmpPerf <- gsub(paste0("\\tval-auc"), "", tmpPerf)
  tmpPerf <- gsub("[[]", "", tmpPerf)
  tmpPerf <- gsub("]", "", tmpPerf)
  tmpPerf <- lapply(tmpPerf, function(x) data.table(t(as.numeric(unlist(strsplit(x, ":"))))))
  tmpPerf <- rbindlist(tmpPerf)
  setnames(tmpPerf, c("nrounds", paste0("val_auc")))
  tmpPerf[, nrounds:=nrounds+1]
  tmpPerf[, cv_index:=j]
  cv_perf_list[[j]] <- tmpPerf
  cv_model_list[[j]] <- tmpFit
  b <- Sys.time()
  print(tmpPerf[which.max(val_auc), ])
  print(difftime(b, a))
}

cv_perf <- rbindlist(cv_perf_list)
cv_error <- cv_perf[, .(mean_auc=mean(val_auc)), by="nrounds"]
setorder(cv_error, -mean_auc)
cv_error
best_nrounds <- cv_error[1, nrounds]
save(cv_perf, param0, best_nrounds, file=paste0(out_dir, "/xgboost_cv.RData"))

# Predict folds using best_nrounds
for(j in 1:5) {
  print(j)
  # Predict
  val_index <- train_df$cv_index == j
  dval <- xgb.DMatrix(data=data.matrix(train_df[val_index, feature_names]), label=train_df[val_index, outcome_name], missing=NA)
  train_df$pred[val_index] <- predict(object=cv_model_list[[j]], newdata=dval, ntreelimit=best_nrounds)
}

# Prepare output
final_cv_pred <- data.table(id=1:nrow(train_df), obs=train_df[, outcome_name], pred=train_df$pred)

#-------------------------------------------------------------------------------
# Fit final model
#-------------------------------------------------------------------------------
# Data
dtrain <- xgb.DMatrix(data=data.matrix(train_df[, feature_names]), label=train_df[, outcome_name], missing=NA)
dtest <- xgb.DMatrix(data=data.matrix(test_df[, feature_names]), missing=NA)

# Fit final model
set.seed(1718)
finalFit <- xgb.train(params=param0, data=dtrain, nrounds=2500, maximize=TRUE)

# Predict test datasets
modelPred <- predict(object=finalFit, newdata=dtest, ntreelimit=best_nrounds)
testPred <- data.table(obs=test_df[, outcome_name], pred=modelPred)
test_AUC <- auc(testPred[, obs], testPred[, pred])

# Find optimal cutoff
cutoff <- seq(0.001, 0.99, by=0.001)
test_acc_diff <- rep(NA, length(cutoff))
for(i in 1:length(cutoff)) {
  # print(i)
  tmp_class <- ifelse(modelPred > cutoff[i], 1, 0)
  confMat <- table(test_df[, outcome_name], tmp_class)
  tmpOut <- try(abs((confMat[1, 1] / sum(confMat[1, ])) - (confMat[2, 2] / sum(confMat[2, ]))))
  if(class(tmpOut) != "try-error" ) test_acc_diff[i] <- tmpOut
}
opt_cutoff <- cutoff[which.min(test_acc_diff)]
testPred[, pred_class:=ifelse(pred > opt_cutoff, 1, 0)]
test_conf_mat <- testPred[, table(obs, pred_class)]
test_acc <- paste0(round(sum(diag(test_conf_mat))*100/sum(test_conf_mat), 2), "%")

# Plot ROC
auc_rf <- roc(response=test_df[, outcome_name], predictor=testPred[, pred])
# plot(auc_rf, print.thres="best", main=paste('AUC:', round(auc_rf$auc[[1]], 3)))
# abline(h=1, col='blue')
# abline(h=0, col='green')
png(filename = paste0(out_dir, "/4. xgboost roc.png"), width=1200, height=680)
ggroc(auc_rf, acc=test_acc)
dev.off()

# Variable Importance
importance_matrix <- xgb.importance(feature_names, model=finalFit)
png(filename = paste0(out_dir, "/4. xgboost var importance.png"), width=1200, height=680)
xgb.plot.importance(importance_matrix[1:15, ])
dev.off()

# Plot tree
xgb.plot.tree(feature_names=feature_names, model=finalFit, n_first_tree=2)

save.image(paste0(out_dir, "/ouput_xgboost.RData"))
