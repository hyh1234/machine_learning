% predictors 
   X = randn(100,5)
   r = [0;2;0;-3;0] % only two nonzero coefficients
   % responses
   Y = X*r + randn(100,1)*.1 % small added noise
   small_sigma_squared = 0.01
   eta_sqaured = 0.01
   bayesian_regression(X,Y,small_sigma_squared,eta_sqaured)