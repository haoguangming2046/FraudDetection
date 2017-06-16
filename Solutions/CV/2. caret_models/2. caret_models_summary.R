#-------------------------------------------------------------------------------
# Consumer Loan Default Prediction
# Data Source    : GitHub h2o
# Problem Type   : Classification
# Total Obs      : 163,987
# Training Obs   : 114,791 (70%)
# Test Obs       :  49,196 (30%)
# Technique      : All caret models
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Environment setup
#-------------------------------------------------------------------------------
# Clear console
rm(list=ls(all=TRUE))
gc()

# Set working directory
setwd("E:/Education/POC/Fraud")
out_dir <- "Solutions/CV/2. caret_models"

# Load libraries
pkgs <- c("data.table", "caret", "R.utils", "doSNOW", "pROC")
sapply(pkgs, require, character.only=TRUE)

#-------------------------------------------------------------------------------
# Caret models (with class probabilities)
#-------------------------------------------------------------------------------
Methods <- c("xgboost", "gbm", "blackboost", "AdaBoost.M1", "ada", "gamSpline", "LMT",
  "multinom", "glm", "bayesglm", "plsRglm", "glmboost", "ctree", "slda",
  "C5.0", "PART", "ctree2", "treebag")

for(i in 1:length(Methods)) {
  load(paste0(out_dir, "/ouput_", Methods[i], ".RData"))
  if(i==1) {
    all_cv_pred <- copy(final_cv_pred)
    setnames(all_cv_pred, "pred", Methods[i])
    all_test_pred <- copy(testPred)
    setnames(all_test_pred, "pred", Methods[i])
  } else {
    all_cv_pred[, c(Methods[i]):=final_cv_pred[, pred]]
    all_test_pred[, c(Methods[i]):=testPred[, pred]]
  }
}
save(all_cv_pred, all_test_pred, file=paste0(out_dir, "/2. caret_models_final_output.RData"))
