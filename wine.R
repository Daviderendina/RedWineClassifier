### PACCHETTI 
install.packages(c("ggpubr", "dplyr", "FactoMineR", "factoextra", "multiROC", 
                   "caret", "MLmetrics"))
install.packages(c("e1071","kernlab", "naivebayes"))
library(e1071)
library(kernlab)
library(naivebayes)
library(multiROC) 
library(FactoMineR) 
library(ggpubr)
library(factoextra) 
library(caret) 
library(dplyr) 

# Diagrams utilities
my.colors <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
legend.dots <- list(space="right", points = list(col=my.colors[1:3]), 
                   text=list(c("LOW","MEDIUM", "HIGH")))
legend.lines <- list(space="right", lines = list(col=my.colors[1:3]), 
                     text=list(c("LOW","MEDIUM", "HIGH")))


# Load data
wine.data = read.csv(paste(dirname(rstudioapi::getSourceEditorContext()$path), 
                           "/winequality-red.csv", sep=""),
                     header = TRUE, sep=",")



#### DATASET EXPLORATION ####

  head(wine.data)
  sapply(wine.data, class)
  dim(wine.data)
  
  # Check no NA cell
  sum(is.na(wine.data))
  
  # Check if values are correct
  str(wine.data)
  
  # Single boxlpots
  par(mfrow=c(1,6)) 
  for(i in 1:11) 
  { boxplot(wine.data[,i], main=names(wine.data)[i]) }
  
  # Dataset balancing
  print_quality_histogram <- function(){
    ggplot(data = wine.data, aes(x = quality, 
                                 y = prop.table(stat(count)),
                                 fill = quality, 
                                 label = scales::percent(prop.table(stat(count))))) +
      geom_bar(position = "dodge") + 
      geom_text(stat = 'count',
                position = position_dodge(.9), 
                vjust = -0.5, 
                size = 3) + 
      scale_y_continuous(labels = scales::percent)
  }
  
  wine.data$quality <- as.factor(wine.data$quality)
  par(mfrow=c(1,1))
  print_quality_histogram()
  
  # Divide target in 3 classes (instead of 10)
  wine.data$quality <- as.character(wine.data$quality)
  wine.data[wine.data$quality %in% as.character(c(1:5)),]$quality <- "LOW"
  wine.data[wine.data$quality == "6",]$quality <- "MEDIUM"
  wine.data[wine.data$quality %in% as.character(c(7:10)),]$quality <- "HIGH"
  wine.factors <- c("LOW", "MEDIUM", "HIGH")
  wine.data$quality <- factor(wine.data$quality, levels = wine.factors)
  
  # Check if levels of the categorical variable are OK
  levels(wine.data$quality)
  
  # Print histogram with new target suddivision
  print_quality_histogram()
  
  # Print boxplot for each variable vs. quality
  par(mfrow=c(3,4)) 
  for(i in 1:11) { 
    boxplot(wine.data[,i] ~ wine.data$quality, main=names(wine.data)[i],
            ylab = names(wine.data)[i], xlab = "quality") 
  }
  
  # Plot scatterplot matrix and density featureplot
  splom(wine.data[1:11], 
        groups=wine.data$quality, 
        col = my.colors, pscales=0,
        key = legend.dots
  )
  
  #Plot density
  featurePlot(x=wine.data[,1:11], y=wine.data[,12], plot="density", 
              scales=list(x=list(relation="free"), y=list(relation="free")), 
              key = legend.lines, col = my.colors,
              layout = c(6,2))


  
  
#### PCA ####
  
  set.seed(123)
  pca.colors <- c("#00AFBB", "#E7B800", "#FC4E07")
  
  # Select all available components, and choose next (TODO si può fare?)
  pca.res <- PCA(wine.data[,1:11], scale.unit = TRUE, graph = FALSE, ncp = 11)
  
  # Show eigenvalues
  get_eigenvalue(pca.res)
  
  # Select only 7 dimension (90% amount of variance)
  wine.pca <- cbind(pca.res$ind$coord[,1:7], wine.data[12])
  
  # Plot variables 
  fviz_pca_var(pca.res, col.var="contrib", gradient.cols = pca.colors)
  
  par(mfrow=c(1,2))
  # Plot individuals over first 2 dimensions
  fviz_pca_ind(pca.res, col.ind = "cos2", 
               geom = "point",
               gradient.cols = pca.colors)
  
  # Plot individuals over first 2 dimensions by quality
  fviz_pca_ind(pca.res, label="none", habillage=wine.data$quality, 
               addEllipses = TRUE,
               palette = my.colors[1:3])
  
    
  
  
#### MACHINE LEARNING MODELS ####
  
  # Split train e test set
  trainIndex <- createDataPartition(wine.pca$quality, p=0.30, list=FALSE)
  wine.train <- wine.pca[trainIndex,] 
  wine.test <- wine.pca[-trainIndex,]
  
  control = trainControl(method = "repeatedcv", number = 10,repeats = 3, 
                         summaryFunction = multiClassSummary, classProbs = TRUE) 
  
  # Rpart
  rpart.model = train(quality ~ ., data = wine.train,metric="AUC" , method = "rpart", trControl = control)
  rpart.res = predict(rpart.model, wine.test[,1:7], type = "prob") %>% mutate('class'=names(.)[apply(., 1, which.max)])
  rpart.pred = factor(rpart.res[,4], levels = wine.factors)
  rpart.probs = rpart.res[,1:3]
  rpart.stats = confusionMatrix(rpart.pred, wine.test$quality, mode="prec_recall") 
  
  # Svm linear
  svml.model = train(quality~ ., data = wine.train, metric="AUC" , method = "svmLinear", trControl = control)
  svml.res = predict(svml.model, wine.test[,1:7], type = "prob") %>% mutate('class'=names(.)[apply(., 1, which.max)])
  svml.pred = factor(svml.res[,4], levels = wine.factors)
  svml.probs = svml.res[,1:3]
  svml.stats = confusionMatrix(svml.pred, wine.test$quality, mode="prec_recall")
  
  # Svm radial
  svmr.model = train(quality~ ., data = wine.train, metric="AUC" , method = "svmRadial", trControl = control)
  svmr.res = predict(svmr.model, wine.test[,1:7], type = "prob") %>% mutate('class'=names(.)[apply(., 1, which.max)])
  svmr.pred = factor(svmr.res[,4], levels = wine.factors)
  svmr.probs = svmr.res[,1:3]
  svmr.stats = confusionMatrix(svmr.pred, wine.test$quality, mode="prec_recall")
  
  # Svm poly
  svmp.model = train(quality~ ., data = wine.train, metric="AUC" , method = "svmPoly", trControl = control)
  svmp.res = predict(svmp.model, wine.test[,1:7], type = "prob") %>% mutate('class'=names(.)[apply(., 1, which.max)])
  svmp.pred = factor(svmp.res[,4], levels = wine.factors)
  svmp.probs = svmp.res[,1:3]
  svmp.stats = confusionMatrix(svmp.pred, wine.test$quality, mode="prec_recall") 
  
  #Naive bayes 
  nb.model = train(quality~ ., data = wine.train, metric="AUC" , method = "naive_bayes", trControl = control)
  nb.res = predict(nb.model, wine.test[,1:7], type = "prob") %>% mutate('class'=names(.)[apply(., 1, which.max)])
  nb.pred = factor(nb.res[,4], levels = wine.factors)
  nb.probs = nb.res[,1:3]
  nb.stats = confusionMatrix(nb.pred, wine.test$quality, mode="prec_recall") 
  
  # Neural net
  nn.model = train(quality~., data=wine.train, type="Classification", metric="AUC" ,method = "nnet",   trControl=control)
  nn.res = predict(nn.model, wine.test[,1:7],type="prob") %>% mutate('class'=names(.)[apply(., 1, which.max)])
  nn.pred = factor(nn.res[,4], levels = wine.factors)
  nn.probs = nn.res[,1:3]
  nn.stats = confusionMatrix(nn.pred, wine.test$quality, mode="prec_recall")
  

  
#### CALCULATE METRICS AND PLOT GRAPHS ####

  plot_ROC <- function(result, name){
    res <- result
    n_method <- length(unique(res$Methods))
    n_group <- length(unique(res$Groups))
    res_df <- data.frame(Specificity= numeric(0), Sensitivity= numeric(0), Group = character(0), AUC = numeric(0), Method = character(0))
    for (i in 1:n_method) {
      for (j in 1:n_group) {
        temp_data_1 <- data.frame(Specificity=res$Specificity[[i]][j],
                                  Sensitivity=res$Sensitivity[[i]][j],
                                  Group=unique(res$Groups)[j],
                                  AUC=res$AUC[[i]][j],
                                  Method = unique(res$Methods)[i])
        colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
        res_df <- rbind(res_df, temp_data_1)
        
      }
      temp_data_2 <- data.frame(Specificity=res$Specificity[[i]][n_group+1],
                                Sensitivity=res$Sensitivity[[i]][n_group+1],
                                Group= "Macro",
                                AUC=res$AUC[[i]][n_group+1],
                                Method = unique(res$Methods)[i])
      temp_data_3 <- data.frame(Specificity=res$Specificity[[i]][n_group+2],
                                Sensitivity=res$Sensitivity[[i]][n_group+2],
                                Group= "Micro",
                                AUC=res$AUC[[i]][n_group+2],
                                Method = unique(res$Methods)[i])
      colnames(temp_data_2) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
      colnames(temp_data_3) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
      res_df <- rbind(res_df, temp_data_2)
      res_df <- rbind(res_df, temp_data_3)
    }
    ggplot2::ggplot(res_df, ggplot2::aes(x = 1-Specificity, y=Sensitivity)) +
      ggplot2::geom_path(ggplot2::aes(color = Group, linetype=Method)) +
      ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), colour='grey', linetype = 'dotdash') +
      ggplot2::theme_bw() +
      ggplot2::ggtitle(name) +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), legend.justification=c(1, 0), legend.position=c(.95, .05), legend.title=ggplot2::element_blank(), legend.background = ggplot2::element_rect(fill=NULL, size=0.5, linetype="solid", colour ="black"))
  }
  create_multiROC_df <- function(modelname, dataframe, model.probs) {
    cnames.low = paste("LOW_pred_",modelname, sep="")
    cnames.med = paste("MEDIUM_pred_",modelname, sep="")
    cnames.high = paste("HIGH_pred_",modelname, sep="")
    df <- cbind(data.frame(matrix(0, ncol = 6, nrow = nrow(dataframe)), dataframe$quality))
    colnames(df) <- c("LOW_true", "MEDIUM_true", "HIGH_true", 
                      cnames.low, cnames.med, cnames.high, "quality")
    df[df$quality == "LOW", "LOW_true"] <- 1
    df[df$quality == "MEDIUM", "MEDIUM_true"] <- 1
    df[df$quality == "HIGH", "HIGH_true"] <- 1
    df[,cnames.low] <- model.probs[,"LOW"] 
    df[,cnames.med] <- model.probs[,"MEDIUM"] 
    df[,cnames.high] <- model.probs[,"HIGH"] 
    df[1:6]
  }
  
  # Create appropriate dataframe and generate multiROC values
  df <- create_multiROC_df("rpart", wine.test, rpart.probs)
  res.rpart <- multi_roc(df, force_diag=T)
  
  df <- create_multiROC_df("svm_linear", wine.test, svml.probs)
  res.svml <- multi_roc(df, force_diag=T)
  
  df <- create_multiROC_df("svm_radial", wine.test, svmr.probs)
  res.svmr <- multi_roc(df, force_diag=T)
  
  df <- create_multiROC_df("svm_polynomial", wine.test, svmp.probs)
  res.svmp <- multi_roc(df, force_diag=T)
  
  df <- create_multiROC_df("naive_bayes", wine.test, nb.probs)
  res.nb <- multi_roc(df, force_diag=T)
  
  df <- create_multiROC_df("nn", wine.test, nn.probs)
  res.nn <- multi_roc(df, force_diag=T)
  
  # Unique dataframe for AUC values
  res.AUC <- cbind(naive_bayes=res.nb$AUC$naive_bayes, 
                   rpart=res.rpart$AUC$rpart,
                   svm_linear=res.svml$AUC$svm_linear,
                   svm_radial = res.svmr$AUC$svm_radial,
                   svm_polynomial = res.svmp$AUC$svm_polynomial,
                   neural_net = res.nn$AUC$nn)
    
  # Print accuracy plot
  d = as.data.frame(rbind(
    c(rpart.stats$overall[c(1,3,4)], "rpart"),
    c(svml.stats$overall[c(1,3,4)], "SVM lineare"),
    c(svmp.stats$overall[c(1,3,4)], "SVM polinomiale"),
    c(svmr.stats$overall[c(1,3,4)], "SVM radiale"),
    c(nb.stats$overall[c(1,3,4)], "naive bayes"),
    c(nn.stats$overall[c(1,3,4)], "neural net")))
  df <- data.frame(x = d$V4,
                   F = as.numeric(d[,1]),
                   L = as.numeric(d[,2]),
                   U = as.numeric(d[,3]))
  
  ggplot(df, aes(x = F, y = x)) +
    geom_point(size = 3) +
    geom_errorbar(aes(xmax = U, xmin = L)) +
    xlim(0,1) + 
    xlab('Accuracy') +
    ylab('') + 
    ggtitle("Valori di accuracy con intervallo di confidenza al 95%")
  
  # Do resample and print timings values
  cv.values = resamples(list(
    svm_linear=svml.model,
    svm_radial=svmr.model,
    svm_polynomial=svmp.model,
    naive_bayes=nb.model,
    rpart = rpart.model,
    neural_net = nn.model)) 
  summary(cv.values)
  
  cv.values$timings
  
  
  ## Metrics
  
  add_stats <- function(df, stats_to_add, name, class.name) {
    stats_to_add <- stats_to_add$byClass[,c("Precision","Recall","F1")]
    f <- as.data.frame(stats_to_add[paste("Class:", class.name),])
    newrows <- data.frame(V1 = f[,1], metric = rownames(f), model = name)
    newds <- merge(newrows, df, all.x=T, all.y=T)
    newds
  }
  create_final_stats <- function(class.name) { 
    t <- as.data.frame(t(as.data.frame(res.AUC[class.name,])))
    t$metric <- "AUC"
    t$model <- rownames(t)
    rownames(t) = 1:nrow(t)
    
    t <- add_stats(t, rpart.stats, "rpart", class.name)
    t <- add_stats(t, svml.stats, "svm_linear", class.name)
    t <- add_stats(t, svmr.stats, "svm_radial", class.name)
    t <- add_stats(t, svmp.stats, "svm_polynomial", class.name)
    t <- add_stats(t, nb.stats, "naive_bayes", class.name)
    t <- add_stats(t, nn.stats, "neural_net", class.name)
    t
  }
  
  # Create unified statistics for each model
  stats.final.low <- create_final_stats("LOW")
  stats.final.medium <- create_final_stats("MEDIUM")
  stats.final.high <- create_final_stats("HIGH")
  
  # Replace NA with 0 
  stats.final.low[is.na(stats.final.low)] <- 0
  stats.final.medium[is.na(stats.final.medium)] <- 0
  stats.final.high[is.na(stats.final.high)] <- 0
  
  #Calculate MACRO avg
  macro.V1 = 
    (stats.final.low[order(stats.final.low[,"metric"], stats.final.low[,"model"]),]$V1 +
       stats.final.medium[order(stats.final.medium[,"metric"], stats.final.medium[,"model"]),]$V1 +
       stats.final.high[order(stats.final.high[,"metric"], stats.final.high[,"model"]),]$V1) / 3
  macro.models = stats.final.high[order(stats.final.high[,"metric"], stats.final.high[,"model"]),]$model
  macro.metrics = stats.final.high[order(stats.final.high[,"metric"], stats.final.high[,"model"]),]$metric
  stats.final.macro <- cbind.data.frame(V1 = macro.V1, metric = macro.metrics, model = macro.models)
  stats.final.macro[is.na(stats.final.macro)] <- 0
  
  ## Plot values
  my.colors.finalplot = c("#38ada9","#b8e994","#6a89cc","#82ccdd","#3c6382","#60a3bc")
  
  plot_final_stats <- function(stats, name){
    ggplot(stats, aes(factor(metric), V1, color=model, fill=model )) + 
      geom_bar(stat="identity", position = "dodge") + 
      scale_color_manual(values = my.colors.finalplot) +
      scale_fill_manual(values = my.colors.finalplot) +
      ggtitle(name) +
      ylim(0,1) +
      ylab("") +
      xlab("")
  }
    

  plot_avg = plot_final_stats(stats.final.macro, "Metriche macro average")
  plot_low = plot_final_stats(stats.final.low, "Metriche per la classe LOW")
  plot_med = plot_final_stats(stats.final.medium, "Metriche per la classe MEDIUM")
  plot_high = plot_final_stats(stats.final.high, "Metriche per la classe HIGH")
  
  ggarrange(plot_avg, plot_low, plot_med, plot_high, nrow=2, ncol=2, common.legend = TRUE, legend="bottom")
  
  
  
  # plot ROC for each model

  plot_ROC(res.rpart, "Curva ROC per rpart")
  plot_ROC(res.svml, "Curva ROC per SVM con kernel lineare")
  plot_ROC(res.svmr, "Curva ROC per SVM con kernel radiale")
  plot_ROC(res.svmp, "Curva ROC per SVM con kernel polinomiale")
  plot_ROC(res.nb, "Curva ROC per naive bayes")
  plot_ROC(res.nn, "Curva ROC per neural net")

  
  
  ## Overlay plot per confronti
  print_conflict_plot <- function(name){
    ggplot() +
      geom_bar(stats1, mapping = aes(factor(metric), V1, group=model), # 10-fold 3-repeated (VERDE)
               stat='identity', position = "dodge", alpha = 0.5, 
               fill = "red", color = "red") +
      geom_bar(stats2, mapping = aes(factor(metric), V1, group=model), # 10 fold normale 56B4E9
               stat='identity', position = "dodge", alpha = 0.5,
               fill = "red", color = "red") +
      ggtitle(name) +
      scale_fill_continue(name="fill legend", label="fill label") +
      scale_fill_discrete(name="fill legend", label="fill label")
  }
  