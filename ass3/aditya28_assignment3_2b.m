clear all;
clc;
numofClass = 3;

%loading data_iris
A = load('data_iris');

%storing the data in X&Y
X = A.X;
Y = A.Y;
[x_row, x_col]= size(X);

%declaring variables with sie
avg_Mu = zeros(numofClass,x_col);
avg_Pi = zeros(numofClass,1);
avg_Sigma =zeros(x_col,x_col,numofClass);
avg_Sigmapooled_LDA = zeros(x_col,1);
limit = 10;
CCR_QDA = zeros(limit,1);
CCR_LDA = zeros(limit,1);
confusionmat_QDA = zeros(numofClass,numofClass,limit);
confusionmat_LDA = zeros(numofClass,numofClass,limit);
avg_Sigmapooled = zeros(x_col,x_col);

%partitioning the data into train and test
for i = 1:limit
    concat = [X Y];
    B = concat(randperm(size(concat,1)),:);
    X_train = B(1:100,1:end-1); 
    Y_train = B(1:100,end);
    X_test = B(101:150,1:end-1);
    Y_test = B(101:150,end);

    %function call to QDA
    [QDAmodel]= aditya28_QDA_train(X_train, Y_train, numofClass);
    [Y_predict_QDA] = aditya28_QDA_test(X_test, QDAmodel, numofClass);
   
    %calculating the sum of all MU,Pi and Sigma
    avg_Mu = avg_Mu + QDAmodel.Mu;
    avg_Pi = avg_Pi + QDAmodel.Pi;
    avg_Sigma = (avg_Sigma + QDAmodel.Sigma);
    
    %function call to LDA
    [LDAmodel]= aditya28_LDA_train(X_train, Y_train, numofClass);
    [Y_predict_LDA] = aditya28_LDA_test(X_test, LDAmodel, numofClass);
    
    avg_Sigmapooled = avg_Sigmapooled + LDAmodel.Sigmapooled;

    %confusionmatrix for QDA &LDA
    confusionmat_QDA(:,:,i) = confusionmat(Y_test,Y_predict_QDA);
    confusionmat_LDA(:,:,i) = confusionmat(Y_test,Y_predict_LDA);
    
   
    %CCR for QDA&LDA
    CCR_QDA(i) = (sum(diag(confusionmat_QDA(:,:,i))))/length(Y_test);
    CCR_LDA(i) = (sum(diag(confusionmat_LDA(:,:,i))))/length(Y_test);
end

%best and worst confusionmatrix for LDA
[worst_ccr,ind1] = min(CCR_LDA);
[best_ccr,ind2] = max(CCR_LDA);
confusionmat_LDA_best = confusionmat_LDA(:,:,ind2);
confusionmat_LDA_worst = confusionmat_LDA(:,:,ind1);


%calculating the averages
avg_Mu = (avg_Mu/i);
avg_Pi = (avg_Pi/i);
avg_Sigma_QDA = (avg_Sigma/i);

%variance for all 4 dimensions for LDA(diagonal of covariance matrix)
avg_Sigmapooled_LDA = (avg_Sigmapooled/i);

%calculating the mean CCR for QDA &LDA
mean_CCR_QDA =mean(CCR_QDA);
mean_CCR_LDA =mean(CCR_LDA);

%calculating the standard deviation for QDA &LDA
std_dev_QDA = std(CCR_QDA);
std_dev_LDA = std(CCR_LDA);

%calculating the variance of all 4 dimensions for each class in QDA
var_matrix = zeros(x_col,numofClass);
for j = 1:3
    var_matrix(:,j) = diag(avg_Sigma(:,:,j)); 
end
        
    














