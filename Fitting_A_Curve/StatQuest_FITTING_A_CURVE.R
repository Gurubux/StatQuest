## first let's make a noisy gamma distribution plot...
x <- seq(from=0, to=20, by=0.1)
y.gamma <- dgamma(x, shape=2, scale=2)
y.gamma.scaled <- y.gamma * 100

y.norm <- vector(length=201)
for (i in 1:201) {
  y.norm[i] <- rnorm(n=1, mean=y.gamma.scaled[i], sd=2) 
}

data <- data.frame(x, y.norm)

plot(data, frame.plot=FALSE, xlab="", ylab="", col="#d95f0e", lwd=1.5)

## Now that we have the data, let's look at the differences 
## and similarities between R's lowess() function and the loess() function.

## We'll start with the lowess() function...
##
## By default "lowess()" fits a line in each window using
## 2/3's of the data points.
##
## the first parameter, y.norm ~ x, says that y.norm is being
## modeled by x, and the second parameter, f, is the fraction
## of points to use in each window. Here, we're using 1/5 of the
## data points in each window.
lo.fit.gamma <- lowess(y.norm ~ x, f=1/5)

plot(data, frame.plot=FALSE, xlab="", ylab="", col="#d95f0e", lwd=1.5)
lines(x, lo.fit.gamma$y, col="black", lwd=3)

## Now use loess() to fit a curve to the data...
##
## By default "loess()" fits a parabola in each window using
## 75% of the data points.
plx<-predict(loess(y.norm ~ x, span=1/5, degree=2, family="symmetric", iterations=4), se=T)

plot(data, frame.plot=FALSE, xlab="", ylab="", col="#d95f0e", lwd=1.5)
lines(x, plx$fit, col="black", lwd=3)

## Now let's add a confidence interval to the loess() fit...
plot(data, type="n", frame.plot=FALSE, xlab="", ylab="", col="#d95f0e", lwd=1.5)
polygon(c(x, rev(x)), c(plx$fit + qt(0.975,plx$df)*plx$se, rev(plx$fit - qt(0.975,plx$df)*plx$se)), col="#99999977")
points(data, col="#d95f0e", lwd=1.5)
lines(x, plx$fit, col="black", lwd=3)

## Now that we know how those functions work... we can skip all that
## nasty stuff and just use ggplot2 with geom_point() and geom_smooth()
library(ggplot2)
ggplot(data=data, aes(x=x, y=y.norm)) +
  geom_point() +
  geom_smooth(span=1/5)
