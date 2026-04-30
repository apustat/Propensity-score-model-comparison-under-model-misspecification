library(mvtnorm)
library(randomForest)   # Random forest
library(MASS)           # LDA
library(e1071)          #for svm

r=1000
eps <- 0.025 #to avoid extreme PS values of 0 or 1
n  <- 200
p <- 9   # number of TRUE confounders (excluding treatment)
mu <- rep(0, p)
rho <- 0.7
Sigma <- rho ^ abs(outer(1:p, 1:p, "-"))

result=matrix(NA, r, 9)
for (i  in 1:r) {
  set.seed(123+i)
  
  X <- rmvnorm(n, mean = mu, sigma = Sigma)
  colnames(X) <- paste0("X", 1:p)  # X1 ... Xp
  X <- as.data.frame(X)
  
  beta <- c(1.2, -0.8, 0.6, 1.1, -0.9, 0.5, 0.7, -1.0, -0.9) 
  
  beta0 <- 0  # intercept
  lin_pred <- beta0 + as.matrix(X) %*% beta
  ps_true <- plogis(lin_pred)
  
  # Treatment variable (0=control, 1=treatment)
  A <- rbinom(n, size = 1, prob = ps_true) #treatment is dependent on covariates
  dat <- data.frame(A = A, X)
  
  alpha0 <- 0          # outcome intercept
  tau    <- 2        # true treatment effect
  gamma  <- c(0.6, -0.4, 1.1, 0.5, -1.2, 0.9, -0.3, -0.8, 1.5)
  
  Y <- alpha0 + tau * dat$A + as.matrix(dat[, paste0("X", 1:p)]) %*% gamma + rnorm(nrow(dat),0,1)
  dat$Y <- Y

  # Build formula for logistic regression
  ps_formula <- as.formula(
    paste("A ~", paste(colnames(X), collapse = " + ")))
  
  #PS model using logistic regression
  ps_lr_fit <- glm(ps_formula, data = dat, family = binomial)
  ps_lr <- predict(ps_lr_fit, type = "response")
  ps_lr <- pmin(pmax(ps_lr, eps), 1 - eps)
  dat$ps_lr <- ps_lr #cor(ps_true, ps_LR)
  
  # Subset data to only treatment + selected covariates
  rf_data <- dat[, c("A", colnames(X))]
  rf_data$A <- as.factor(rf_data$A)  # treat A as categorical
  # Fit Random Forest directly with the formula object
  ps_rf_fit <- randomForest(
    formula = ps_formula,  # <- pass the formula object directly
    data = rf_data,
    ntree = 500)
  ps_rf <- predict(ps_rf_fit, type = "prob")[, 2]  # probability of A=1
  ps_rf <- pmin(pmax(ps_rf, eps), 1 - eps)
  dat$ps_rf <- ps_rf
  
  #PS model using LDA
  ps_lda_fit <- lda(
    formula = ps_formula,
    data = dat)
  
  ps_lda <- predict(ps_lda_fit)$posterior[, 2]  # probability of A=1
  ps_lda <- pmin(pmax(ps_lda, eps), 1 - eps)
  dat$ps_lda <- ps_lda
  
  #PS model using SVM
  ps_svm_fit <- svm(formula = ps_formula, data = dat, kernel  = "linear",   # or "radial", "polynomial"
                    probability = TRUE)
  ps_svm <- predict(ps_svm_fit,
                    newdata = transform(dat, A = factor(A, levels = c(0, 1))),
                    probability = TRUE)
  
  ps_svm <- pmin(pmax(ps_svm, eps), 1 - eps)
  dat$ps_svm <- ps_svm
  
  # Logistic regression PS
  dat$w_lr <- ifelse(dat$A == 1, 1/dat$ps_lr, 1/(1-dat$ps_lr))
  
  # Random forest PS
  dat$w_rf <- ifelse(dat$A == 1, 1/dat$ps_rf, 1/(1-dat$ps_rf))
  
  # LDA PS
  dat$w_lda <- ifelse(dat$A == 1, 1/dat$ps_lda, 1/(1-dat$ps_lda))
  
  # SVM PS
  dat$w_svm <- ifelse(dat$A == 1, 1/dat$ps_svm, 1/(1-dat$ps_svm))
  
  #IPW estimator
  ipw <- function(A, Y, ps) {
    mean(A*Y/ps - (1-A)*Y/(1-ps))
  }
  ipw_lr = ipw(A = dat$A, Y = dat$Y , ps = dat$ps_lr)
  ipw_rf = ipw(A = dat$A, Y = dat$Y , ps = dat$ps_rf)
  ipw_lda = ipw(A = dat$A, Y = dat$Y , ps = dat$ps_lda)
  ipw_svm = ipw(A = dat$A, Y = dat$Y , ps = dat$ps_svm)
  
  # AIPW estimator
  # Fit outcome model
  outcome_vars <- union("A", colnames(X)) #adding treatment variable in the outcome model if it's not selected by any chance
  outcome_formula <- as.formula(
    paste("Y ~", paste(outcome_vars, collapse = " + ")))
  outcome_fit <- lm(outcome_formula, data = dat)
  
  # Predict potential outcomes
  f1 <- predict(outcome_fit, newdata = transform(dat, A=1))
  f0 <- predict(outcome_fit, newdata = transform(dat, A=0))
  
  aipw <- function(A, Y, ps, f1, f0) {
    mean(
      A * (Y - f1) / ps -
        (1 - A) * (Y - f0) / (1 - ps) +
        (f1 - f0)
    )
  }
  aipw_lr=aipw(A=dat$A, Y=dat$Y, ps=dat$ps_lr, f1=f1, f0=f0)
  aipw_rf=aipw(A=dat$A, Y=dat$Y, ps=dat$ps_rf, f1=f1, f0=f0)
  aipw_lda=aipw(A=dat$A, Y=dat$Y, ps=dat$ps_lda, f1=f1, f0=f0)
  aipw_svm=aipw(A=dat$A, Y=dat$Y, ps=dat$ps_svm, f1=f1, f0=f0)
  
  # regression estimate/Response Surface Model (RSM)
  RSM=mean(f1 - f0)
  
  result[i, 1]=ipw_lr
  result[i, 2]=ipw_rf
  result[i, 3]=ipw_lda
  result[i, 4]=ipw_svm
  
  result[i, 5]=aipw_lr
  result[i, 6]=aipw_rf
  result[i, 7]=aipw_lda
  result[i, 8]=aipw_svm
  result[i, 9]=RSM
}
colnames(result)<-c("IPW-LR", "IPW-RF", "IPW-LDA", "IPW-SVM",
                    "AIPW-LR", "AIPW-RF", "AIPW-LDA", "AIPW-SVM",
                    "RSM")
apply(result,2,mean)

write.csv(result, file="C:/Users/apust/OneDrive - University of Nebraska Medical Center/Desktop/PS comparison/1.n=200-rho=0.7.csv")
