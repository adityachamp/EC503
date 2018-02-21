clear all;
clc;
A = load('data_cancer');
gamma = 0.1:0.05:1;
t = 1;
len_gamma = length(gamma);
X = A.X;
Y = A.Y;
numofClass = 2;
cm_RDA_Xtest = zeros(numofClass,numofClass,len_gamma);
cm_RDA_Xtrain = zeros(numofClass,numofClass,len_gamma);

ccr_X_test = zeros(len_gamma,1);
ccr_X_train = zeros(len_gamma,1);

for gamma = 0.1:0.05:1
concat = [X Y];
    Q = concat(randperm(size(concat,1)),:);
    X_train = Q(1:150,1:end-1);
    Y_train = Q(1:150,end);
    X_test = Q(151:216,1:end-1);
    Y_test = Q(151:216,end);
    [x_row, x_col]= size(X_test);
    [x1_row,x1_col] = size(X_train);
    
%passing train data with gamma and numofClass
[RDAmodel]= aditya28_RDA_train(X_train, Y_train,gamma, numofClass);

%calculating confusionmatrix and CCR for Xtest
[Y_predict] = aditya28_RDA_test(X_test, RDAmodel, numofClass);
cm_RDA_Xtest = confusionmat(double(Y_test),Y_predict-1);
ccr_X_test(t) = sum(diag(cm_RDA_Xtest))/x_row;

%calculating confusionmatrix and CCR for Xtrain
[Y_predict] = aditya28_RDA_test(X_train, RDAmodel, numofClass);
cm_RDA_Xtrain = confusionmat(double(Y_train),Y_predict-1);
ccr_X_train(t) = sum(diag(cm_RDA_Xtrain))/x1_row;
t = t+1;
end

%plotting X_test & X_train vs gamma
gamma = 0.1:0.05:1;
plot(gamma,ccr_X_test,'o')
xlabel('gamma')
ylabel('ccrXtest')
title('Plot of X test and X train vs Gamma')
hold on
plot(gamma,ccr_X_train,'p')
xlabel('gamma')
ylabel('ccrXtrain and ccrXtest')
hold off
legend('ccrXtest','ccrXtrain')