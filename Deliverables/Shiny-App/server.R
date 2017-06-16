library(data.table)
library(ggplot2)
library(xgboost)
library(Ckmeans.1d.dp)
library(shiny)
library(dplyr)
library(pROC)

load("data/Data.RData")
load("data/Output.RData")
importance_matrix <- importance_matrix[1:15, ]
source("xgb.importance.plot.R")
source("ggroc.R")

test_dt[, pred_prob:=modelPred]
test_dt[, pred_class:=ifelse(pred_prob > opt_cutoff, 1, 0)]
setorder(test_dt, -bad_loan, -pred_class)
outData <- test_dt[, list(id, int_rate, annual_inc, dti, loan_amnt, term, bad_loan, pred_prob, pred_class)]
outData[, pred_class:=as.integer(pred_class)]
outData[, annual_inc:=as.integer(annual_inc)]
outData[, term:=as.integer(term)]
outData[, pred_prob:=round(pred_prob, 3)]
confMat <- outData[, table(bad_loan, pred_class)]
confMat <- data.table(data.frame(confMat))
confMat[, Percetage:=paste0(round(Freq/sum(Freq)*100, 2), "%")]
setnames(confMat, c("Bad Loan", "Predicted Class", "Frequency", "Percetage"))
setnames(outData, c("id", "int_rate", "annual_inc", "dti", "loan_amnt", "term", "bad_loan", "pred_prob",
  "pred_class"), c("ID", "Interest Rate", "Annual Income", "Debt to Income Ratio", "Loan Amount",
  "Tenure", "Bad Loan", "Predicted Probabilty", "Predicted Class"))
rawData[, bad_loan:=factor(bad_loan)]
# Define server logic required to draw a histogram
shinyServer(
  function(input, output) {
    # Generate a summary of the data
    output$str <- renderPrint({
      summary(rawData)
    })
    output$importance <- renderPlot({
      xgb.importance.plot(importance_matrix)
    })
    output$roc <- renderPlot({
      ggroc(ROC, test_acc)
    })
    output$model <- renderPlot({
      xgb.plot.tree(feature_names=feature_names, model=finalFit, n_first_tree=1)
    })
    output$view <- renderDataTable(outData, options=list(pageLength = 10, dom="ltipr"))
    output$conf <- renderDataTable(confMat, options=list(dom="t"))
  }
)