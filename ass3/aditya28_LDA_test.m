 function [Y_predict] = aditya28_LDA_test(X_test, LDAmodel, numofClass);
%
% Testing for LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% LDAmodel : the parameters of LDA classifier which has the follwoing fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your codes here:

%testing LDA by direct sustitution into he formula
for i = 1:length(X_test)
    for j = 1:numofClass
       LDA(i,j) =((LDAmodel.Mu(j,:))*(inv(LDAmodel.Sigmapooled)))*(X_test(i,:)') - (0.5*LDAmodel.Mu(j,:))*(inv(LDAmodel.Sigmapooled))*((LDAmodel.Mu(j,:))') + (log(LDAmodel.Pi(j,:)));
    end
end
%calculating argmax
[value1,Y_predict] = max(LDA, [], 2);

end
