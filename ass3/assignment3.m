%QDA train
load('data_iris');
concat = [X Y];
B = concat(randperm(size(concat,1)),:);
x_train = B(1:100,1:end-1);
y_train = B(1:100,end);
x_test = B(101:150,end-1);
y_test = B(101:150,end);
test = [x_train y_train];
numofclass = 3;
%first_row = test(1,:);
class1 = [];
class2 = [];
class3 = [];
for i=1:100
    if y_train(i) == 1
        class1 = [class1;x_train(i,:)];
    elseif y_train(i) == 2
        class2 = [class2;x_train(i,:)];
    elseif y_train(i) == 3
        class3 = [class3;x_train(i,:)];
    end
end
cov_c1 = cov(class1);
cov_c2 = cov(class2);
cov_c3 = cov(class3);
mean_1 = mean(class1, 1);
mean_2 = mean(class2, 1);
mean_3 = mean(class3, 1);
mean_matrix = [mean_1;mean_2;mean_3];
prob_matrix = [numel(class1)/numel(x_train);numel(class2)/numel(x_train);numel(class3)/numel(x_train)];
        
    














