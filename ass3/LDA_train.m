function [LDAmodel] = LDA_train(X_train, Y_train, numofClass)
%
% Training LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% LDAmodel : the parameters of LDA classifier which has the following fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%

% Write your codes here:


end
