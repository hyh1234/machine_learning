function CALL_bayesian_regression

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - CALL_bayesian_regression
% Creation Date - 3rd Nov 2014
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description: 
%   Function to do Bayesian regression
%   inspired by video on bayesian linear regression
%   https://www.youtube.com/watch?v=qz2U8coNwV4
%   by mathematicalmonk on youtube
%
% Input:  
%
% Output: 
%       1) Vector of inferred regressors/parameters  
%       2) Histograms of inferred regressors/parameters
%
% Assumptions -
%
% Example usage:
%           CALL_bayesian_regression
%
% License - BSD 
%
% Acknowledgements -
%           Dedicated to my wife Joyeeta Ghose and my mother Kalyani
%           Banerjee
%
% Change History - 
%                   3rd Nov 2014  - Creation by Soumya Banerjee
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


   % predictors 
    X = randn(100,1)
    r=[0.2]
   %X = randn(100,5)
   %r = [0;2;0;-3;0] % only two nonzero coefficients
   % responses
   Y = X*r + randn(100,1)*.1 % small added noise
   small_sigma_squared = 0.01
   eta_sqaured = 0.01
   plot(X,Y,'.');
   bayesian_regression(X,Y,small_sigma_squared,eta_sqaured)