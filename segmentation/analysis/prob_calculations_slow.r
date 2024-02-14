library("mvtnorm")
library("cubature")

inv_x <- function(x) {
  return(1 / (x %*% x))
}

calculate_t <- function(delta, k, w, y) {

  x1 <- seq(delta, k * delta, by = delta)
  x2 <- seq(0, (w - 1) * delta, by = delta)

  a <- inv_x(x1) * x1

  b <- c(-(k + 1) * delta * sum(inv_x(x2) * x2) * a, inv_x(x2) * x2)

  beta_1 <- as.double(a %*% y[c(1:k)])

  beta_2 <- as.double(b %*% y[c(1:(k + w))])

  y_hat_2 <- rep(delta * (k + 1) * beta_1, w) + beta_2 * x2

  var_2 <- var(y_hat_2 - y[c((k + 1):(k + w))])

  if (var_2 == 0) {
    return(Inf)
  } else {
    t <- as.double((beta_2 - beta_1) / sqrt((var_2 * inv_x(x2)) / w))

    return(t)
  }

}

density_to_int <- function(a, delta, k, w, mu, cov, y) {
  t <- calculate_t(delta, k, w, y[c(1:(k + w))])

  t_ <- calculate_t(delta, k + 1, w, y)

  return(as.integer((abs(t) > abs(t_)) & (abs(t) > a)) * dmvnorm(y, mu, cov))
}

prob_success <- function(a, delta, k, w, mu, cov, lower, upper) {

  f <- function(y) density_to_int(a, delta, k, w, mu, cov, y)

  return(cubintegrate(f = f, lower = lower, upper = upper, maxEval = 10^5))
}

sig_level <- 0.99

delta <- 1 / 90
k <- 5
w <- 5

x <- seq(delta, (k+w) * delta, by = delta)

beta1 <- -0.5
beta2 <- 0.5

c2 <- (beta1-beta2)*(k+1)*delta

mu <- c(beta1 * x[c(1:k)], beta2 * x[c((k + 1):(k + w))] + c2, beta2 * (k + w + 1) * delta + c2)

var1 <- 0.00001
var2 <- 0.00005

var <- c(rep(var1, k), rep(var2, w + 1))
cov <- diag(var)

y <- c(4.00000, 3.00000,  2.00000 , 1.02846, 11.00000, 20.98500, 31.00000, 41.00000)

a <- qt(sig_level, w - 2)

lower <- mu - 3 * sqrt(var)
upper <- mu + 3 * sqrt(var)

print(prob_success(a, delta, k, w, mu, cov, lower, upper))
