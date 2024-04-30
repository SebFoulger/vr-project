library('segmented')

df_data <- read.csv('D:\\Uni\\Project\\repo\\vr-project\\input_data\\controller\\speed\\1_1_1.csv')

plot(df_data$controller_speed_clean[c(1:400)])
start.time <- Sys.time()
selgmented(df_data$controller_speed_clean[c(1:4000)], Kmax=400, type='bic', stop.if = 401,9 th=10, G=16, check.dslope=FALSE, bonferroni = FALSE)
end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)