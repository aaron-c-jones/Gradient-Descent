### Gradient Descent Example ----
require(gridExtra)
require(ggplot2)

##Loss Function 1 and Derivative
lossFunc <- function(x){ 2.9 * (x - 7)^2 + 1.5 }
deriv <- function(x){ 2.9 * 2 * (x-7) }

##Algorithm
gradientDescent <- function(init, max_it = 1000, tol_err = 1e-06, learning_rate = 0.2, adaptive = TRUE){
  it = 0
  x = init
  learningRate = learning_rate
  relErr = 1
  
  xDf = numeric(); xDf[1] = x
  stop = 0
  while(stop == 0 & it < max_it){
    it = it + 1
    
    gradient = deriv(x)
    xNew = x - as.numeric(learningRate)*gradient
    xDf[it + 1] = xNew
    
    cat('it = ', it, '\n', 'x = ', xNew, '\n', 'learning rate = ', learningRate, '\n')
    cat('------------------------------', '\n')
    
    relErr = abs(x - xNew) / max(1, abs(xNew))
    if(relErr <= tol_err) stop = 1
    
    if(adaptive == TRUE){
      ##Barzilai and Borwein - Adaptive Learning Rate
      deltaX = xNew - x
      deltaG = deriv(xNew) - gradient
      learningRate = (t(deltaG) %*% deltaX) / (t(deltaG) %*% deltaG)
    }
    
    x = xNew
  }
  df = data.frame(x = xDf, y = lossFunc(xDf))
  return(df)
}

x0 = 5
withAdapt = gradientDescent(x0)
withoutAdapt = gradientDescent(x0, adaptive = FALSE)

##Plotting
#Given Adaptive Learning Rate
segmentWithAdapt <- data.frame(x = double(), y = double(), xend = double(), yend = double())
for(i in 1:(nrow(withAdapt) - 1)){
  segmentWithAdapt[i, ] <- cbind(withAdapt[i, ], withAdapt[i + 1, ])
}
plot1withAdapt <- {ggplot(data.frame(x = c(5, 9)), aes(x)) + 
    stat_function(fun = lossFunc) + 
    geom_point(data = withAdapt, aes(x, y), color = 'red', size = 3, shape = 2) +
    geom_segment(data = segmentWithAdapt, aes(x = x, y = y, xend = xend, yend = yend), color = 'blue', arrow = arrow(length = unit(0.2, 'cm'))) +
    labs(title = 'Path of Convergence (With Adaptive Learning Rate)', x = 'Input', y = 'Cost') +
    theme_bw()}
plot2withAdapt <- {ggplot(withAdapt, aes(rownames(withAdapt), y, group = 1)) +
    geom_line() + 
    labs(title = paste0('# of Iterations = ', max(rownames(withAdapt)), ' (Including Initial Value)'), x = 'Iteration', y = '') +
    theme_bw()}
grid.arrange(plot1withAdapt, plot2withAdapt, ncol = 2)

#Given No Adaptive Learning Rate
segmentWithoutAdapt <- data.frame(x = double(), y = double(), xend = double(), yend = double())
for(i in 1:(nrow(withoutAdapt) - 1)){
  segmentWithoutAdapt[i, ] <- cbind(withoutAdapt[i, ], withoutAdapt[i + 1, ])
}
plot1withoutAdapt <- {ggplot(data.frame(x = c(5, 9)), aes(x)) + 
    stat_function(fun = lossFunc) + 
    geom_point(data = withoutAdapt, aes(x, y), color = 'red', size = 3, shape = 2) +
    geom_segment(data = segmentWithoutAdapt, aes(x = x, y = y, xend = xend, yend = yend), color = 'blue', arrow = arrow(length = unit(0.2, 'cm'))) +
    labs(title = 'Path of Convergence (Without Adaptive Learning Rate)', x = 'Input', y = 'Cost') +
    theme_bw()}
plot2withoutAdapt <- {ggplot(withoutAdapt, aes(rownames(withoutAdapt), y, group = 1)) +
    geom_line() + 
    labs(title = paste0('# of Iterations = ', max(rownames(withoutAdapt)), ' (Including Initial Value)'), x = 'Iteration', y = '') +
    theme_bw()}
grid.arrange(plot1withoutAdapt, plot2withoutAdapt, ncol = 2)
