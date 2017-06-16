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
out_dir <- "Solutions/CV/3. caret_models"

# Load libraries
pkgs <- c("data.table", "caret", "R.utils", "doSNOW")
sapply(pkgs, require, character.only=TRUE)

# Record start time
startTime <- Sys.time()

# Set number of cross-validation folds
nFolds <- 4

#-------------------------------------------------------------------------------
# Data Processing
#-------------------------------------------------------------------------------
# Import dataset
load("Solutions/CV/2. caret_models/2. caret_models_final_output.RData")
all_cv_pred[, id:=NULL]
setnames(all_cv_pred, "obs", "bad_loan")
setnames(all_test_pred, "obs", "bad_loan")

# Convert outcome variable to factor
all_cv_pred[, bad_loan:=ifelse(bad_loan==0, "No", "Yes")]
all_cv_pred[, bad_loan:=factor(bad_loan)]

all_test_pred[, bad_loan:=ifelse(bad_loan==0, "No", "Yes")]
all_test_pred[, bad_loan:=factor(bad_loan)]

# Id variables
outcome_name <- "bad_loan"
feature_names <- setdiff(names(all_cv_pred), outcome_name)

# Convert data.table into data.frame
train_df <- data.frame(all_cv_pred)
test_df <- data.frame(all_test_pred)

# Create cross-validation folds
set.seed(1718)
train_df[, "cv_index"] <- createFolds(train_df[, outcome_name], k=nFolds, list=FALSE)

# Remove unwanted data from the session
rm(list=setdiff(ls(), c("out_dir", "startTime", "nFolds", "outcome_name", "feature_names", "train_df", "test_df")))
gc()

#-------------------------------------------------------------------------------
# Function for finding cross-validation error for one iteration of parameters
#-------------------------------------------------------------------------------
paramIter <- function(id_var, X, Y, nFolds, cvIndex, Method, tuneGrid, myControl, pkgs) {
  # Record process start time
  startTime <- Sys.time()

  # Cross-validation for one row of parameters
  cv_pred_list <- foreach(k=1:nFolds, .inorder=FALSE, .errorhandling="pass", .packages=c("caret", "data.table", pkgs)) %dopar% {
    # Model fit
    set.seed(1718)
    modelFit <- caret::train(x = X[cvIndex != k, ]
      , y = Y[cvIndex != k]
      , method = Method
      , tuneGrid = tuneGrid
      , trControl = myControl
      , metric = "ROC"
    )

    # Predict fold holdout datasets
    modelPred <- predict(modelFit, newdata=X[cvIndex == k, ], type="prob")
    out <- data.table(id=id_var[cvIndex == k], obs=Y[cvIndex == k], pred=modelPred[, "Yes"])
    out
  }

  # Prepare final outputs
  cv_pred <- rbindlist(cv_pred_list)

  # Model performance
  library(pROC)
  AUC <- auc(cv_pred[, obs], cv_pred[, pred])

  # Record process end time
  endTime <- Sys.time()
  timeTaken <- round(as.numeric(difftime(endTime, startTime, units="sec")))

  # Return output
  out <- list(cv_pred=cv_pred, timeTaken=timeTaken, AUC=AUC)
  return(out)
}

#-------------------------------------------------------------------------------
# Caret models (with class probabilities)
#-------------------------------------------------------------------------------
Methods <- c("ada", "AdaBag", "adaboost", "AdaBoost.M1", "amdai", "avNNet",
  "awnb", "awtan", "bag", "bagEarth", "bagEarthGCV", "bagFDA", "bagFDAGCV",
  "bam", "bartMachine", "bayesglm", "bdk", "binda", "blackboost", "Boruta",
  "C5.0", "C5.0Rules", "C5.0Tree", "cforest", "chaid", "ctree", "ctree2",
  "dda", "dnn", "dwdLinear", "dwdPoly", "dwdRadial", "earth", "evtree",
  "extraTrees", "fda", "gam", "gamboost", "gamLoess", "gamSpline",
  "gaussprLinear", "gaussprPoly", "gaussprRadial", "gbm", "gcvEarth", "glm",
  "glmboost", "glmnet", "glmStepAIC", "gpls", "hda", "hdda", "hdrda", "J48",
  "JRip", "kernelpls", "kknn", "knn", "lda", "lda2", "Linda", "LMT", "loclda",
  "logicBag", "LogitBoost", "logreg", "manb", "mda", "mlp", "mlpML",
  "mlpWeightDecay", "mlpWeightDecayML", "multinom", "nb", "nbDiscrete",
  "nbSearch", "nnet", "nodeHarvest", "oblique.tree", "OneR", "ordinalNet",
  "ORFlog", "ORFpls", "ORFridge", "ORFsvm", "pam", "parRF", "PART", "pcaNNet",
  "pda", "pda2", "plr", "pls", "plsRglm", "polr", "qda", "QdaCov", "randomGLM",
  "ranger", "rbf", "rbfDDA", "Rborist", "rda", "rf", "rlda", "rmda",
  "rotationForest", "rotationForestCp", "rpart", "rpart1SE", "rpart2", "RRF",
  "RRFglobal", "rrlda", "sda", "sddaLDA", "sddaQDA", "sdwd", "simpls", "slda",
  "sparseLDA", "spls", "stepLDA", "stepQDA", "svmBoundrangeString", "svmExpoString",
  "svmLinear", "svmLinear2", "svmLinearWeights", "svmPoly", "svmRadial",
  "svmRadialCost", "svmRadialSigma", "svmRadialWeights", "svmSpectrumString", "tan",
  "tanSearch", "treebag", "vbmpRadial", "vglmAdjCat", "vglmContRatio", "vglmCumulative",
  "widekernelpls", "wsrf", "xgbLinear", "xgbTree", "xyf")

# Training Control
myControl <- trainControl(method = "none"
  , verboseIter = FALSE
  , returnData = FALSE
  , returnResamp = "none"
  , savePredictions = FALSE
  , classProbs = TRUE
  , summaryFunction = twoClassSummary
)

#-------------------------------------------------------------------------------
# Loop for different techniques
#-------------------------------------------------------------------------------
i <- which(Methods=="xgbTree")

for(i in 1:length(Methods)) {
  # Print technique currently running
  cat("===========================================================================\n")
  cat(Methods[i], "\n")

  # List of packages that need to be loaded
  pkgs <- getModelInfo(Methods[i])[[Methods[i]]]$library
  # install.packages(pkgs)
  sapply(pkgs, require, character.only=TRUE)

  # List of parameters that need to be tuned
  param_list <- param_out <- try(getModelInfo(Methods[i])[[Methods[i]]]$grid(x=train_df[, feature_names], y=train_df[, outcome_name], len=3))
  if(class(param_out) == "try-error") next
  param_list <- data.table(param_list)
  print(param_list)

  # Store prediction
  cv_pred_list <- list()

  #-------------------------------------------------------------------------------
  # Loop for selection of parameters
  #-------------------------------------------------------------------------------
  for(j in 1:nrow(param_list)) {
    # Register clusters
    cl <- snow::makeCluster(5, type="SOCK")
    doSNOW::registerDoSNOW(cl)

    # Tuning grid
    tuneGrid <- data.frame(param_list[j, ])
    if("parameter" %in% names(tuneGrid)) tuneGrid <- NULL

    # Find the cross-validation accuracy
    tmpOut <- try(withTimeout(paramIter(id_var=1:nrow(train_df), X=train_df[, feature_names], Y=train_df[, outcome_name], nFolds,
      cvIndex=train_df$cv_index, Method=Methods[i], tuneGrid, myControl, pkgs), timeout=1800))

    # Stop clusters
    # system("ps -A")
    # system("killall -KILL R")
    system("Taskkill /IM Rscript.exe /F", show.output.on.console=FALSE)

    # Store final output
    if(class(tmpOut) != "try-error") {
      cv_pred_list[[j]] <- tmpOut$cv_pred
      cv_pred_list[[j]][, iteration:=j]
      param_out$AUC[j] <- tmpOut$AUC
      param_out$timeTaken[j] <- tmpOut$timeTaken
      param_out$iteration[j] <- j
    } else {
      cat(j, " - Error in current iteration\n")
      break
    }

     # Print output
     print(param_out[j, ])
  }

  param_out <- data.table(param_out)
  save(cv_pred_list, param_out, file=paste0(out_dir, "/", Methods[i], ".RData"))
  cat("Max AUC: ", max(param_out$AUC), "\n")
}
