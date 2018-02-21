function [Y_predict] = aditya28_QDA_test(X_test, QDAmodel, numofClass);
% Testing for QDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% QDAmodel: the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance
% matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% 
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your code here:

%testing QDA by direct sustitution into he formula
for i = 1:length(X_test)
    for j = 1:numofClass
        QDA(i,j) = (((X_test(i,:)-QDAmodel.Mu(j,:))*(0.5*(inv(QDAmodel.Sigma(:,:,j)))))*((X_test(i,:)-QDAmodel.Mu(j,:))'))+(0.5*log(det(QDAmodel.Sigma(:,:,j)))-log(QDAmodel.Pi(j,:)));
    end
%calculating argmin
[value,Y_predict] = min(QDA, [], 2);
end
end
