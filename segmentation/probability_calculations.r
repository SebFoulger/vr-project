inv_x <- function(x) {
  return(1 / (x %*% x))
}

prob_t_sig <- function(delta, mu, var, val, k, w) { # create a function with the name my_function

  x1 = seq(delta, k * delta, by = delta)
  x2 = seq(0, (w-1) * delta, by = delta)

  var <- var[c(1:(k + w))]
  mu <- mu[c(1:(k + w))]

  a <- inv_x(x1) * x1

  b <- c(-(k+1) * delta * sum(inv_x(x2)*x2) * a, inv_x(x2)*x2)

  d <- c(b[c(1:k)]-a, b[c((k+1):(k+w))]) * sqrt((w * (w - 1)) / ((val ^ 2) * inv_x(x2)))

  e <- x2 %o% b

  e <- e + cbind(matrix(-(k+1)*delta*a, nrow=w, ncol=k, byrow=TRUE),diag(1, nrow = w, ncol = w))
  
  M <- d %o% d

  for (i in 1:(k+w)) {
    for (j in 1:(k+w)) {
      M[i, j] <- M[i, j] - (e[, i] %*% e[, j])
    }
  }

  ev <- eigen(diag(var^0.5) %*% M %*% diag(var^0.5))
  lambda <- ev$values
  P <- ev$vectors

  b2 <- t(P) %*% diag(var^-0.5) %*% mu

  return(imhof(q = 0, lambda = lambda, delta = abs(b2), epsrel = 10^(-7)))
}

k = 10
w = 10

delta=1/90

x = seq(delta, (k+w) * delta, by = delta)

beta1 = -0.5
beta2 = 0.5

c2 = (beta1-beta2)*(k+1)*delta

mu = c(beta1*x[c(1:k)], beta2*x[c((k+1):(k+w))]+c2)

var1 = 0.00001
var2 = 0.00005

var = c(rep(var1, k),rep(var2,w + 1))

sig_level = 0.9999

print(1-prob_t_sig(delta=delta, mu = mu, var = var, val = qt(sig_level,w-2), k = k, w = w)$Qq)
