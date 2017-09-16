### Linear Model Gradient Descent Example ----
require(gridExtra)
require(ggplot2)

attach(mtcars)

fit <- lm(hp ~ wt, data = mtcars)
plotOrig <- {ggplot(mtcars, aes(wt, hp)) +
  geom_point(size = 3) +
  geom_abline(aes(intercept = coef(fit)[1], slope = coef(fit)[2], colour = 'blue'), size = 1.1) +
  labs(title = 'Linear Model', x = 'Disp', y = 'MPG') +
  theme_bw()}
plotOrig + scale_colour_manual(name = '', values = c('blue' = 'blue'), labels = c('Via Closed-Form Solution'))

##Loss Function 1 and Derivative
lossFunc <- function(y, yhat, n){ (1 / n) * sum((y - yhat)^2) }
derivb0 <- function(y, yhat, n){ (1 / n) * sum(2 * (yhat - y)) }
derivb1 <- function(x, y, yhat, n){ (1 / n) * sum(2 * (yhat - y) * x) }

##Algorithm
gradientDescent <- function(x, y, nparams, max_it = 1000000, tol_err = 1e-08, learning_rate = 0.001, adaptive = TRUE){
  it = 0
  
  n = length(x)
  b0 = runif(1, 0, 1)
  b1 = runif(1, 0, 1)
  yhat = b0 + b1 * x
  mse = sum((y - yhat)^2) / n
  
  learningRateb0 = learning_rate
  learningRateb1 = learning_rate
  
  dfb0 = numeric(); dfb0[1] = b0
  dfb1 = numeric(); dfb1[1] = b1
  dfmse = numeric(); dfmse[1] = mse
  
  stop = 0
  while(stop == 0 & it < max_it){
    it = it + 1
    
    b0New = b0 - as.numeric(learningRateb0) * derivb0(y, yhat, n)
    b1New = b1 - as.numeric(learningRateb1) * derivb1(x, y, yhat, n)
    yhatNew = b0New + b1New * x
    mseNew = sum((y - yhatNew)^2) / n
    
    dfb0[it+1] = b0New
    dfb1[it+1] = b1New
    dfmse[it+1] = mseNew
    
    cat('it = ', it, '\n',
        'b0, b1 = ', b0New, ',', b1New, '\n',
        'mse = ', mseNew, '\n',
        'learning rate b0, learning rate b1 = ', learningRateb0, ',', learningRateb1, '\n')
    cat('--------------------------------------------------', '\n')
    
    error = abs(mse - mseNew)
    if(error <= tol_err) stop = 1
    
    if(adaptive == TRUE){
      ##Barzilai and Borwein - Adaptive Learning Rate
      deltab0 = b0New - b0
      deltaGb0 = derivb0(y, yhatNew, n) - derivb0(y, yhat, n)
      learningRateb0 = (t(deltaGb0) %*% deltab0) / (t(deltaGb0) %*% deltaGb0) 
      
      deltab1 = b1New - b1
      deltaGb1 = derivb1(x, y, yhatNew, n) - derivb1(x, y, yhat, n)
      learningRateb1 = (t(deltaGb1) %*% deltab1) / (t(deltaGb1) %*% deltaGb1)
    }
    
    b0 = b0New
    b1 = b1New
    yhat = yhatNew
    mse = mseNew
  }
  
  df = data.frame(b0 = dfb0, b1 = dfb1, mse = dfmse)
  return(df)
}

withAdapt = gradientDescent(x = mtcars$wt, y = mtcars$hp, nparams = length(coef(fit)))
#withoutAdapt = gradientDescent(x = mtcars$wt, y = mtcars$hp, nparams = length(coef(fit)), adaptive = FALSE)

actualCoef = coef(fit)
actualCoef

##Plotting
coefFromGD <-c(withAdapt$b0[nrow(withAdapt)], withAdapt$b1[nrow(withAdapt)])
plotNew <- {plotOrig +
    geom_abline(aes(intercept = coefFromGD[1], slope = coefFromGD[2], colour = 'red'), size = 1.1, linetype = 2) +
    scale_colour_manual(name = '', values = c('blue' = 'blue', 'red' = 'red'), labels = c('Via Closed-Form Solution', 'Via Gradient Descent'))}
plotNew



