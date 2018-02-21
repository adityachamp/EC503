clear all;
clc;
%load data
A = load('data_mnist_train');
B = load('data_mnist_test');

%declaring the final distance matrix
concat_matrix = zeros(60000,10000);

%dividing data into 6 splits of 10000 each
upperlimit = 10000;
lowerlimit = 1;
dist = zeros(10000,10000,6);

%calculating the distance between test and train points
for i = 1:6
%dist(:,:,i) = (B.X_test(:,:)*(B.X_test(:,:)')) - (2*(A.X_train(lowerlimit:upperlimit,:)*(B.X_test(:,:)'))) + ((A.X_train(lowerlimit:upperlimit,:)*(A.X_train(lowerlimit:upperlimit,:)')));
%dist(:,:,i) =  (-2*(A.X_train(lowerlimit:upperlimit,:)*(B.X_test(:,:)'))) + ((A.X_train(lowerlimit:upperlimit,:)*(A.X_train(lowerlimit:upperlimit,:)')));
    
    %vector multiplication
    dist(:,:,i) = (-2*(A.X_train(lowerlimit:upperlimit,:)*(B.X_test(:,:)')))+ diag((A.X_train(lowerlimit:upperlimit,:)*(A.X_train(lowerlimit:upperlimit,:)')));
    upperlimit = upperlimit + 10000;
    lowerlimit =lowerlimit + 10000;
    concat_matrix((10000*(i-1)+1):10000*i,:) = (dist(:,:,i));
end

%calculating CCR and confusion matrix
[value,class] = min(concat_matrix);
class_index = A.Y_train(class);
cm=confusionmat(B.Y_test,class_index);
ccr = (sum(diag(cm))/10000)



    

    