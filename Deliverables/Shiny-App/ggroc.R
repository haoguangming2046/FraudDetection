# Function to plot ROC
ggroc <- function(ROC, acc) {
  interval <- 0.2
  breaks <- seq(0, 1, interval)

  plotx <- rev(ROC$specificities)
  ploty <- rev(ROC$sensitivities)
  sub.title <- paste0("AUC: ", round(ROC$auc, 3), "  Accuracy: ", acc)

  ggplot(NULL, aes(x = plotx, y = ploty)) +
    geom_segment(aes(x = 0, y = 1, xend = 1,yend = 0), colour="red", alpha = 0.5, size=1) +
    geom_step(size=1, colour="blue") +
    scale_x_reverse(name = "Specificity",limits = c(1, 0), breaks = breaks, expand = c(0.001, 0.001)) +
    scale_y_continuous(name = "Sensitivity", limits = c(0, 1), breaks = breaks, expand = c(0.001, 0.001)) +
    theme(axis.ticks = element_line(color = "grey80")) +
    coord_equal() +
    ggtitle(bquote(atop(.("ROC"), atop(italic(.(sub.title)), "")))) +
	theme(plot.title = ggplot2::element_text(lineheight = 0.9, face = "bold"), 
      panel.grid.major.y = ggplot2::element_blank(), text = element_text(size=14, face="bold"))
}
