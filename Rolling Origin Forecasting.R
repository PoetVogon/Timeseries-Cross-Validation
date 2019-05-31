## Time series cross-validation 
## Variation of Rolling Origin Method by Rob J Hyndman
## Link:https://bit.ly/2DJMxJO | https://bit.ly/2UPs1uo

rm(list = ls()) # removes all objects from the environment
cat("\014")     # clears the console

## --------------------------------------------------------------##

library('forecast')
library('fpp')
library('dygraphs')
library('fpp2')
library('tseries')
library('TSA')
library('ggplot2')
library('scales')
library('tseries')
library('caret')
library('tsoutliers')

## ------------------------------------------------------------------------------------- ##
## Import data ##
## ------------------------------------------------------------------------------------- ##

beer = read.csv(choose.files(), header = TRUE)

## Check for missing values or NA values
beer[!complete.cases(beer),]  ## No missing values
cbind(colSums(is.na(beer)))       ## No NA values

## ------------------------------------------------------------------------------------- ##
## Converting the dataset in to a time series object ##
## ------------------------------------------------------------------------------------- ##

tsbeer = ts(beer[, 1], start = c(2001, 1), frequency = 4)

## ------------------------------------------------------------------------------------- ##
## Descriptive statistics and interactive plots ##
## ------------------------------------------------------------------------------------- ##
summary(tsbeer)   ## Average quarterly beer sales ~ 330

## Check if the series has outliers
tso(
  tsbeer, delta = 0.7, types = c("AO", "LS", "TC"),
  maxit = 1, maxit.iloop = 4, maxit.oloop = 4,
  cval.reduce = 0.14286,discard.method = c("en-masse", "bottom-up"),
  tsmethod = c("auto.arima", "arima"), check.rank = FALSE
)

## Interactive graph with date range selector
dygraph(tsbeer, main = "Beer Sales") %>%
  dyAxis("y", label = "Sales (#)",
         valueRange = c(100, 600)) %>%
  dyOptions(axisLineWidth = 2,
            fillGraph = TRUE,
            drawGrid = TRUE) %>%
  dyRangeSelector()

## Highlight good and bad periods
dygraph(tsbeer, main = "Beer Sales",
        xlab = "Year",
        ylab = "Sales") %>%
  dySeries(label = "Sales (#)", color = "blue") %>%
  dyShading(from = "2001-1-1", to = "2003-06-30", color = "#CCEBD6") %>%
  dyShading(from = "2003-07-1", to = "2006-06-30", color = "#ff7f7f") %>%
  dyShading(from = "2006-07-1", to = "2012-03-31", color = "#CCEBD6") %>%
  dyShading(from = "2012-04-1", to = "2013-03-31", color = "#ff7f7f") %>%
  dyShading(from = "2013-04-1", to = "2018-12-31", color = "#CCEBD6") %>%
  dyAnnotation("2002-1-1", text = "P", tooltip = "Steady Growth") %>%
  dyAnnotation("2005-1-1", text = "E", tooltip = "No Growth") %>%
  dyAnnotation("2009-1-1", text = "P", tooltip = "Steady Growth") %>%
  dyAnnotation("2013-1-1", text = "N", tooltip = "Decline") %>%
  dyAnnotation("2016-1-1", text = "P", tooltip = "Steady Growth") %>%
  dyRangeSelector()

## Seasonal plot 01
ggseasonplot(tsbeer, 
             year.labels = TRUE, 
             year.labels.left = TRUE,labelgap = .05) +
  ylab("Sales(#)") +
  ggtitle("Seasonal Plot")

## Seasonal plot 02
ggseasonplot(tsbeer, polar = TRUE) +
  ylab("degree") +
  ggtitle("Polar seasonal plot: Beer Sales")

## Sub-series plot - Quarterly
monthplot(tsbeer)
abline(h = mean(tsbeer))  ## Average quarterly beer sales line

## Dickey-Fuller test of stationarity 
## Ho = Series is non-stationary
## Ha = Series is stationary
adf.test(tsbeer, k=4)

## P-value > 0.05. Cannot reject null hypothesis
## Series is non-stationary

## -------------------------------------------------------------------- ##
## Decomposition to understand trend and seasonality
## -------------------------------------------------------------------- ##
tsdec = decompose(tsbeer, type = "additive") 
plot(tsdec, col = "blue4" )

plot(decompose(tsdec$trend, type = "multiplicative"))
## No seasonality

plot(decompose(tsdec$seasonal, type = "additive"))
## No trend

plot(decompose(tsdec$random, type = "additive"))
## Noise has no significant trend or seasonality

## LJung-Box test
model = auto.arima(tsbeer)
Box.test(model$residuals, lag = 40, type = "Ljung-Box")

## Density plot of residuals
hist(model$residuals,
     col = "blue",
     xlab = "error",
     main = "Density plot of residuals",
     freq = FALSE)
lines(density(model$residuals))
## Residuals are concentrated near 0
## Normally distributed 

## Observing the seasonal indices of "TSDecompose" 
sind = round(t(tsdec$figure), 2) 
colnames(sind) = c("Q1","Q2","Q3","Q4")
sind

## Graph of trend component after decompose 
dygraph(tsdec$trend,
        main = "Trend",
        xlab = "Year",
        ylab = "Sales") %>%
  dySeries(label = "Base Sales (#)",
           color = "blue") %>%
  dyRangeSelector()


## Graph of seasonal component after decompose
dygraph(tsdec$seasonal,
        main = "Seasonal",
        xlab = "Year",
        ylab = "Sales") %>%
  dySeries(label = "Base Sales (#)",
           color = "blue") %>%
  dyRangeSelector()

## Graph of irregular component after decompose
dygraph(tsdec$random,
        main = "Random",
        xlab = "Year",
        ylab = "Sales") %>%
  dySeries(label = "Base Sales (#)",
           color = "blue") %>%
  dyRangeSelector()

## Autocorrelation and Partial Autocorrelation plots
acf(tsbeer, lag.max = 24, type = "correlation", plot = TRUE, 
    na.action = na.fail, demean = TRUE, drop.lag.0 = TRUE)

pacf(tsbeer, lag.max = NULL, plot = TRUE, na.action = na.fail)

forecast::ndiffs(tsbeer)  ## Arima [p,1,q][P,D,Q]
forecast::nsdiffs(tsbeer) ## Arima [p,1,q][P,1,Q]
BoxCox.lambda(tsbeer) ## 0.363166


## ------------------------------------------------------------------------------------- ##
## Model 04 : Rolling Origin Cross-Validation ##
## ------------------------------------------------------------------------------------- ##

l = 60 # minimum  number of quarters for fitting a model (affects accuracy)
m = length(tsbeer)
mat1 = matrix(NA, m - l, 8)   ## Matrix to store MAE values for Arima
mat3 = matrix(NA, m - 1, 8)   ## Matrix to store MAPE values for Arima
mat2 = matrix(NA, m - l, 8)   ## Matrix to store MAE values for Holt-Winters
mat4 = matrix(NA, m - 1, 8)   ## Matrix to store MAPE values for Holt-Winters

tt = tsp(tsbeer)[1] + (l - 2)/4

## model = auto.arima(tsbeer)
## Box.test(model$residuals, lag = 40, type = "Ljung-Box")
## (hw(tsbeer, seasonal = "multiplicative"))$model

for (a in 1:(m - l + 1))
{
  yshort  = window(tsbeer, start = start(tsbeer)[1]+(a-1)*.25, end = tt + (2 * a) / 8, extend = TRUE)
  ynext   = window(tsbeer, start = (tt + (2 * a + 2) / 8), end = (tt + (2 * a + 17) / 8), extend = TRUE)
  fitArima = Arima(
    yshort,
    order = c(1, 1, 1),
    seasonal = list(order = c(0, 1, 1), period = 8),
    include.drift = FALSE,
    lambda = 0,
    method = "ML"
  )
  fcastArima = forecast(fitArima, h = 8)
  if(a<=(m-l)){
    mat1[a, 1:length(ynext)] = abs(fcastArima[['mean']] - ynext)
    mat3[a, 1:length(ynext)] = (abs(fcastArima[['mean']] - ynext))/ynext
  }
  fitHW =  hw(
    yshort,
    seasonal = "multiplicative",
    alpha = 0.0907,
    beta  = 0.0471,
    gamma = 0.0001,
    h = 8
  )
  fcastHW = forecast(fitHW, h = 8)
  if(a<=(m-l)){
    mat2[a, 1:length(ynext)] = abs(fcastHW[['mean']] - ynext)
    mat4[a, 1:length(ynext)] = (abs(fcastHW[['mean']] - ynext))/ynext
  }
  
  print(start(yshort))  ## Training set starting year-quarter
  print(end(yshort))    ## Training set ending year-quarter
  print(start(ynext))   ## Testing set starting year-quarter
  print(end(ynext))     ## Testing set ending year-quarter
  print(round(fcastArima$mean))
}

## MAE Values
mean(mat1, na.rm=TRUE)  ## MAE value of Arima
mean(mat2, na.rm=TRUE)  ## MAE value of Holt-Winters

## MAPE Values
mean(mat3, na.rm=TRUE)  ## MAPE value of Arima
mean(mat4, na.rm=TRUE)  ## MAPE value of Holt-Winters

## Plot the MAE values for Arima and Holt-Winters
dev.new(width=1920, height=1080, unit="pixels")
plot(
  1:8,
  colMeans(mat1, na.rm = TRUE),
  type = "l",
  col = 4,
  xlab = "Two Year Quarters",
  ylab = "Mean Absolute Error",
  ylim = c(4,10),
  lwd = 3
)

lines(1:8,
      colMeans(mat2, na.rm = TRUE),
      type = "l",
      col = 3,
      lwd = 3)

legend(
  "topleft",
  legend = c("Arima", "Holt-Winters"),
  col = 3:4,
  lty = 1,
  lwd = 3,
  cex = 1.5
)

## Plot the MAPE values for Arima and Holt-Winters
dev.new(width=1920, height=1080, unit="pixels")
plot(
  1:8,
  colMeans(mat3, na.rm = TRUE),
  type = "l",
  col = 4,
  xlab = "Two Year Quarters",
  ylab = "Mean Absolute Error",
  ylim = c(0.010, 0.020),
  lwd = 3
)

lines(1:8,
      colMeans(mat4, na.rm = TRUE),
      type = "l",
      col = 3,
      lwd = 3)

legend(
  "topleft",
  legend = c("Arima", "Holt-Winters"),
  col = 3:4,
  lty = 1,
  lwd = 3,
  cex = 1.5
)

## Forecast 2019-Q1 to 2020-Q4
fcastArima = forecast(fitArima, h = 8)  ## Forecast using Arima
fcastHW = forecast(fitHW, h = 8)        ## Forecast using Holt-Winters

## Forecast values
round(fcastArima$mean, digits=2)
round(fcastHW$mean, digits=2)

## Plots of forecast
ts.plot(fcastArima,
        col = c("blue", "green", "green", "black", "black"),
        main = "Beer sales: Forecast for next 8 quarters using Arima")

ts.plot(fcastHW,
        col = c("blue", "green", "green", "black", "black"),
        main = "Beer sales: Forecast for next 8 quarters using Holt-Winters")
