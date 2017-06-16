xgb.importance.plot <- function(importance_matrix = NULL, numberOfClusters = 5) {
  importance_matrix <- importance_matrix[, .(Gain = sum(Gain)), by = Feature]
  clusters <- suppressWarnings(Ckmeans.1d.dp::Ckmeans.1d.dp(importance_matrix[, Gain], numberOfClusters))
  importance_matrix[, `:=`("Cluster", clusters$cluster %>% 
                             as.character)]
  plot <- ggplot2::ggplot(importance_matrix,
    ggplot2::aes(x = stats::reorder(Feature, Gain),
    y = Gain, width = 0.05), environment = environment()) + 
    ggplot2::geom_bar(ggplot2::aes(fill = Cluster), stat = "identity", 
                      position = "identity") + ggplot2::coord_flip() + 
    ggplot2::xlab("Features") + ggplot2::ylab("Gain") + ggplot2::ggtitle("Feature importance") + 
    ggplot2::theme(plot.title = ggplot2::element_text(lineheight = 0.9, face = "bold"), 
                   panel.grid.major.y = ggplot2::element_blank(), text = element_text(size=14, face = "bold"))
                   # ,axis.text.y = element_text(angle=22)
  return(plot)
}

