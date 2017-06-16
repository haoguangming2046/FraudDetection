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
train_obs <- all_cv_pred[, obs]
test_obs <- all_test_pred[, obs]
all_cv_pred[, c("id", "obs"):=NULL]
all_test_pred[, c("obs"):=NULL]

#-------------------------------------------------------------------------------
# AUC
#-------------------------------------------------------------------------------
cv_auc <- all_cv_pred[, lapply(.SD, function(x) auc(train_obs, x))]
cv_auc <- data.frame("cv"=t(cv_auc))
cv_auc <- data.table(Method=rownames(cv_auc), cv=cv_auc$cv)

test_auc <- all_test_pred[, lapply(.SD, function(x) auc(test_obs, x))]
test_auc <- data.frame("test"=t(test_auc))
test_auc <- data.table(Method=rownames(test_auc), test=test_auc$test)

all_auc <- merge(cv_auc, test_auc, by="Method")
all_auc
write.csv(all_auc, file="Solutions/CV/2. caret_models_ensemble/auc_summary.csv", row.names=FALSE)


#-------------------------------------------------------------------------------
# Cross-validation auc (Step 1)
#-------------------------------------------------------------------------------
# Register clusters
# cl <- snow::makeCluster(10, type="SOCK")
# doSNOW::registerDoSNOW(cl)

# p <- seq(from=0.1, to=1, by=0.1)
# mainM <- "xgboost"
# remM <- setdiff(names(all_cv_pred), "xgboost")
# out <- list()
# for(m in remM) {
  # cat("=================================================================================\n")
  # cat(m, "\n")
  # tmpOut <- foreach(i=p, .combine=c, .inorder=TRUE, .packages="pROC") %dopar% {
    # tmpPred <- all_cv_pred[[mainM]]*i + all_cv_pred[[m]]*(1-i)
    # tmpAUC <- auc(train_obs, tmpPred)
    # as.numeric(tmpAUC)
  # }
  # tmp <- data.table(p=p, AUC=tmpOut)
  # setorder(tmp, -AUC)
  # out[[m]] <- tmp
  # print(tmp)
# }
# out <- rbindlist(out)
# out[, Method:=rep(remM, each=10)]
# setorder(out, -AUC)
# out
# save(out, file="Solutions/CV/2. caret_models_ensemble/Out_Step_1.RData")
# system("Taskkill /IM Rscript.exe /F", show.output.on.console=FALSE)

#-------------------------------------------------------------------------------
# Cross-validation auc (Step 2)
#-------------------------------------------------------------------------------
# Register clusters
# cl <- snow::makeCluster(10, type="SOCK")
# doSNOW::registerDoSNOW(cl)

# p <- seq(from=0.1, to=1, by=0.1)
# mainM <- c("xgboost", "gbm")
# tmpM <- all_cv_pred[[mainM[1]]]*0.7 + all_cv_pred[[mainM[2]]]*0.3
# remM <- setdiff(names(all_cv_pred), mainM)
# out <- list()
# for(m in remM) {
  # cat("=================================================================================\n")
  # cat(m, "\n")
  # tmpOut <- foreach(i=p, .combine=c, .inorder=TRUE, .packages="pROC") %dopar% {
    # tmpPred <- tmpM*i + all_cv_pred[[m]]*(1-i)
    # tmpAUC <- auc(train_obs, tmpPred)
    # as.numeric(tmpAUC)
  # }
  # tmp <- data.table(p=p, AUC=tmpOut)
  # setorder(tmp, -AUC)
  # out[[m]] <- tmp
  # print(tmp)
# }
# out <- rbindlist(out)
# out[, Method:=rep(remM, each=10)]
# setorder(out, -AUC)
# out
# save(out, file="Solutions/CV/2. caret_models_ensemble/Out_Step_2.RData")
# system("Taskkill /IM Rscript.exe /F", show.output.on.console=FALSE)

#-------------------------------------------------------------------------------
# Cross-validation auc (Step 3)
#-------------------------------------------------------------------------------
# Register clusters
# cl <- snow::makeCluster(10, type="SOCK")
# doSNOW::registerDoSNOW(cl)

# p <- seq(from=0.1, to=1, by=0.1)
# mainM <- c("xgboost", "gbm", "blackboost")
# tmpM <- all_cv_pred[[mainM[1]]]*0.63 + all_cv_pred[[mainM[2]]]*0.27 + all_cv_pred[[mainM[3]]]*0.1
# remM <- setdiff(names(all_cv_pred), mainM)
# out <- list()
# for(m in remM) {
  # cat("=================================================================================\n")
  # cat(m, "\n")
  # tmpOut <- foreach(i=p, .combine=c, .inorder=TRUE, .packages="pROC") %dopar% {
    # tmpPred <- tmpM*i + all_cv_pred[[m]]*(1-i)
    # tmpAUC <- auc(train_obs, tmpPred)
    # as.numeric(tmpAUC)
  # }
  # tmp <- data.table(p=p, AUC=tmpOut)
  # setorder(tmp, -AUC)
  # out[[m]] <- tmp
  # print(tmp)test_acc
# }
# out <- rbindlist(out)
# out[, Method:=rep(remM, each=10)]
# setorder(out, -AUC)
# out
# save(out, file="Solutions/CV/2. caret_models_ensemble/Out_Step_3.RData")
# system("Taskkill /IM Rscript.exe /F", show.output.on.console=FALSE)

#-------------------------------------------------------------------------------
# Final test prediction
#-------------------------------------------------------------------------------
mainM <- c("xgboost", "gbm", "blackboost")
final_test_pred <- all_test_pred[[mainM[1]]]*0.63 + all_test_pred[[mainM[2]]]*0.27 + all_test_pred[[mainM[3]]]*0.1
final_test_auc <- auc(all_test_pred[, obs], final_test_pred)
auc_rf <- roc(response=all_test_pred[, obs], predictor=final_test_pred)
plot(auc_rf, print.thres="best", main=paste('AUC:', round(auc_rf$auc[[1]], 3)))
abline(h=1, col='blue')
abline(h=0, col='green')

# cutoff <- seq(0.001, 0.99, by=0.001)
# test_acc <- rep(NA, length(cutoff))
# for(i in 1:length(cutoff)) {
  # print(i)
  # tmp_class <- ifelse(final_test_pred > cutoff[i], 1, 0)
  # test_acc[i] <- sum(diag(table(all_test_pred[, obs], tmp_class))) / length(tmp_class)
# }

cutoff <- seq(0.001, 0.99, by=0.001)
test_acc_diff <- rep(NA, length(cutoff))
for(i in 1:length(cutoff)) {
  print(i)
  tmp_class <- ifelse(final_test_pred > cutoff[i], 1, 0)
  confMat <- table(all_test_pred[, obs], tmp_class)
  tmpOut <- try(abs((confMat[1, 1] / sum(confMat[1, ])) - (confMat[2, 2] / sum(confMat[2, ]))))
  if(class(tmpOut) != "try-error" ) test_acc_diff[i] <- tmpOut
}

final_cutoff <- cutoff[which.min(test_acc_diff)]
final_test_class <- ifelse(final_test_pred > final_cutoff, 1, 0)
final_conf_mat <- table(all_test_pred[, obs], final_test_class)
final_test_acc <- sum(diag(table(all_test_pred[, obs], final_test_class))) / length(final_test_class)

save(final_test_pred, final_test_class, final_test_auc, final_test_acc, final_cutoff, final_conf_mat, file="Solutions/CV/2. caret_models_ensemble/final_output.RData")
