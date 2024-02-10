library("CompQuadForm")

inv_x <- function(x) {
  return(1 / as.double(x %*% x))
}

# Calcultes the probability that one t-statistic beats the next
prob_t_beats_t <- function(delta, mu, var, k, w, ratio, positive) {

  x1 <- seq(delta, k * delta, by = delta)
  x2 <- seq(0, (w - 1) * delta, by = delta)

  x1_ <- seq(delta, (k + 1) * delta, by = delta)
  x2_ <- x2

  a <- inv_x(x1) * x1
  a_ <- inv_x(x1_) * x1_

  b <- c(-(k + 1) * delta * sum(inv_x(x2) * x2) * a, inv_x(x2) * x2)
  b_ <- c(-(k + 2) * delta * sum(inv_x(x2_) * x2_) * a_, inv_x(x2_) * x2_)

  if (positive) {
    d <- ratio * (b[c(1:k)] - a[c(1:k)]) - b_[c(1:k)] + a_[c(1:k)]
    d <- c(d, ratio * b[k + 1] - b_[k + 1] + a_[k + 1])
    d <- c(d, ratio * b[c((k + 2):(k + w))] - b_[c((k + 2):(k + w))])
    d <- c(d, -b_[k + w + 1])
  } else {
    d <- ratio * (b[c(1:k)] - a[c(1:k)]) + b_[c(1:k)] - a_[c(1:k)]
    d <- c(d, ratio * b[k + 1] + b_[k + 1] - a_[k + 1])
    d <- c(d, ratio * b[c((k + 2):(k + w))] + b_[c((k + 2):(k + w))])
    d <- c(d, -b_[k + w + 1])
  }
  
  return(1 - pnorm(0, mean = d %*% mu, sd = d %*% (d * var)))
}

ratio_prob <- function(delta, mu, var, k, w, ratio) {

  x1 <- seq(delta, k * delta, by = delta)
  x2 <- seq(0, (w - 1) * delta, by = delta)

  x1_ <- seq(delta, (k + 1) * delta, by = delta)
  x2_ <- x2

  a <- inv_x(x1) * x1
  a_ <- inv_x(x1_) * x1_

  b <- c(-(k + 1) * delta * sum(inv_x(x2) * x2) * a, inv_x(x2)*x2)
  b_ <- c(-(k + 2) * delta * sum(inv_x(x2_) * x2_) * a_, inv_x(x2_) * x2_)

  e <- diag(1, w, k + w) - x2 %o% b

  for (j in (1:k)) {
    for (i in (1:w)) {
      e[i, j] <- e[i, j] - (k + 1) * delta * a[j]
    }
  }

  e_ <- diag(1, w, k + w + 1) - x2_ %o% b_

  for (j in (1:(k + 1))) {
    for (i in (1:w)) {
      e_[i, j] <- e_[i, j] - (k + 2) * delta * a_[j]
    }
  }

  M <- matrix(0, k + w + 1, k + w + 1)

  for (i in (1:(k + w))) {
    for (j in (1:(k + w))) {
      M[i, j] <- M[i, j] - (e[, i] %*% e[, j]) * ratio**2
    }
  }

  for (i in (1:(k + w + 1))) {
    for (j in (1:(k + w + 1))) {
      M[i, j] <- M[i, j] + (e_[, i] %*% e_[, j])
    }
  }

  ev <- eigen(x = diag(var^0.5) %*% M %*% diag(var^0.5), symmetric = TRUE)
  lambda <- ev$values
  P <- ev$vectors

  b2 <- t(P) %*% diag(var^-0.5) %*% mu

  return(1 - imhof(q = 0, lambda = lambda, delta = abs(b2), epsrel = 10^(-7))$Qq)
}

prob_t_beats_t_abs <- function(delta, mu, var, k, w, ratio) {

  return_val <- prob_t_beats_t(delta, mu, var, k, w, ratio, TRUE) 
  return_val <- return_val - (1 - prob_t_beats_t(delta, mu, var, k, w, ratio, FALSE))

  ratio_p <- ratio_prob(delta, mu, var, k, w, ratio)

  return_val <- return_val * ratio_p + 1 - ratio_p

  return(return_val)
}

break_early_prob <- function(k, w, n, delta, beta1, beta2, var1, var2, sig_level, break_tol) {

  x <- seq(delta, (n + w) * delta, by = delta)

  c2 <- (beta1 - beta2) * x[n + 1]

  mu <- c(beta1 * x[c(1:n)], beta2 * x[c((n + 1):(n + w))] + c2)

  var <- c(rep(var1, n), rep(var2, w))

  total_prob <- 0

  max_ratio <- 3
  no_mins <- 72

  for (m in (n - k - w + 3):(n - k - break_tol)) {
    input_mu <- mu[c(m:(m + k + w))] - (m - 1) * beta1 * delta

    f <- function(x) prob_t_beats_t_abs(delta, input_mu, var[c(m:(m + k + w))], k, w, x)

    cur_min <- 1

    # To ensure that we don't get stuck at a local minimum we do multiple intervals
    for (l in 0:no_mins) {
      lower <- (max_ratio / no_mins) * l
      o = optimize(f, interval = c(lower, lower + max_ratio / no_mins), maximum = FALSE)
      if (o$objective < cur_min) {
        cur_min <- max(o$objective, 0)
        best_ratio <- o$minimum
      }
    }
    total_prob <- total_prob + cur_min
  }

  total_prob <- total_prob + sig_level * (n + 2 - k - w - break_tol)

  return (total_prob)
}

print(break_early_prob(10, 10, 30, 1/90, -0.1, 0.1, 0.00001, 0.00001, 0.001, 0))