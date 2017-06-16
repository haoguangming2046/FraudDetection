library(shiny)

style = 
# Define UI for application that draws a histogram
shinyUI(fluidPage(theme = "ciberstyle.css",
  headerPanel(""),
  sidebarLayout(
    sidebarPanel(
      h1("Fraud Detection"),
      br(),
      br(),
      img(src = "fraud.jpg", width=280, height=400),
      br(),
      br(),
      br(),
      img(src = "ciber.png"),
      width = 3
    ),
    mainPanel(
      tabsetPanel(type = "tabs", 
                  tabPanel("Data", verbatimTextOutput("str")), 
                  tabPanel("Model", img(src = "model.png"), width = "400", height = "400"),
                  tabPanel("Output", dataTableOutput("view")),
                  tabPanel("Feature Importance", plotOutput("importance", width = "100%", height = "600px")),
                  tabPanel("Confusion Matrix", dataTableOutput("conf")),
                  tabPanel("ROC", plotOutput("roc", width = "100%", height = "600px"))

      ),
      width=9
    )
  )
))
