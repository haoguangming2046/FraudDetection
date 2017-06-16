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
out_dir <- "Solutions/CV/1. caret_models"

# Load libraries
pkgs <- c("data.table", "caret", "R.utils", "doSNOW")
sapply(pkgs, require, character.only=TRUE)

# Record start time
startTime <- Sys.time()

#-------------------------------------------------------------------------------
# Caret model summary
#-------------------------------------------------------------------------------
modelFiles <- list.files(path = out_dir, pattern = ".RData")
model_acc <- list()
model_param <- list()

for(File in modelFiles) {
  cat(File, "\n")
  tmpFile <- paste0(out_dir, "/", File)
  load(tmpFile)
  if(!is.null(param_out) & !is.null(param_out$AUC)) {
    tmp_param <- param_out[which.max(AUC), ]
    tmp_acc <- tmp_param[, list(AUC, timeTaken)]
    tmp_param[, c("AUC", "timeTaken", "iteration"):=NULL]
    model_acc[[File]] <- tmp_acc
    model_param[[File]] <- tmp_param
  }
  param_out <- NULL
}

final_acc <- rbindlist(model_acc)
final_acc[, model:=names(model_acc)]
final_acc[, model:=gsub(".RData", "", model)]
setorder(final_acc, -AUC)
final_acc <- final_acc[AUC > 0.65 & timeTaken < 500, ]

#-------------------------------------------------------------------------------
# Caret model summary
#-------------------------------------------------------------------------------
modelFiles <- paste0(final_acc[, model], ".RData")
model_param <- list()

for(File in modelFiles) {
  cat(File, "\n")
  tmpFile <- paste0(out_dir, "/", File)
  load(tmpFile)
  if(!is.null(param_out) & !is.null(param_out$AUC)) {
    # setorder(param_out, -AUC)
    model_param[[File]] <- param_out
  }
  param_out <- NULL
}

final_params <- list()
final_params[["gbm"]] <- expand.grid(interaction.depth=1:3, n.trees=seq(50, 500, 50), shrinkage=0.1, n.minobsinnode=10)
final_params[["blackboost"]] <- expand.grid(maxdepth=1:3, mstop=seq(50, 500, 50))
final_params[["AdaBoost.M1"]] <- expand.grid(maxdepth=1:3, mfinal=seq(50, 500, 50), coeflearn="Freund")
final_params[["ada"]] <- expand.grid(maxdepth=1:3, iter=seq(50, 500, 50), nu=0.1)
final_params[["gamSpline"]] <- expand.grid(df=1:10)
final_params[["LMT"]] <- expand.grid(iter=c(1, 11, 21, 31, 41, 51))
final_params[["multinom"]] <- expand.grid(decay=c(0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6))
final_params[["glm"]] <- expand.grid(parameter="none")
final_params[["bayesglm"]] <- expand.grid(parameter="none")
final_params[["plsRglm"]] <- expand.grid(nt=1:10, alpha.pvals.expli=0.01)
final_params[["glmboost"]] <- expand.grid(mstop=seq(50, 500, 50), prune=c("no", "yes"))
final_params[["ctree"]] <- expand.grid(mincriterion=seq(0.01, 0.99, length=10))
final_params[["slda"]] <- expand.grid(parameter="none")
final_params[["PART"]] <- expand.grid(threshold=0.25, pruned=c("yes"))
final_params[["C5.0"]] <- expand.grid(trials=c(1, 10, 20, 30, 40), model=c("tree", "rules"), winnow=FALSE)
final_params[["ctree2"]] <- expand.grid(maxdepth=1:10, mincriterion=0.01)
final_params[["treebag"]] <- expand.grid(parameter="none")
save(final_params, file="Solutions/CV/2. caret_models/final_params.RData")

