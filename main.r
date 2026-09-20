library(tseries)
library(rugarch)


spx <- read.csv("spx.csv", header = FALSE)
log.returns <- diff(log(spx$V2))


qqnorm(log.returns, main = "Normal Q-Q Plot: SPX Log Returns")
qqline(log.returns, col = "blue", lwd = 2)

jarque.bera.test(log.returns)


acf(log.returns^2, main = "ACF: Squared SPX Log Returns")
Box.test(log.returns^2, lag = 10, type = "Ljung-Box")


spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                   mean.model = list(armaOrder = c(0,0)))
fit <- ugarchfit(spec = spec, data = log.returns)
fit


Rt_lagged <- log.returns[-length(log.returns)]
abs_after_pos <- abs(log.returns[-1][Rt_lagged > 0])
abs_after_neg <- abs(log.returns[-1][Rt_lagged < 0])

cor(log.returns[-1]^2, Rt_lagged)

qqplot(abs_after_pos, abs_after_neg,
       xlab = "Quantiles of |Rt| after positive returns",
       ylab = "Quantiles of |Rt| after negative returns")
abline(0, 1, col = "blue", lwd = 2)
