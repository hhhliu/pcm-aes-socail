function [x,y,auc] = rocauc(predict,real)
pos_num = sum(real==1);
neg_num = sum(real==0);

m = size(real,1);
[pre,index] = sort(predict,'descend');
ground_truth = real(index);
x = zeros(m+1,1);
y = zeros(m+1,1);
auc = 0;
x(1) = 0;
y(1) = 0;

for i=2:m
    tp = sum(ground_truth(1:i)==1);
    fp = sum(ground_truth(1:i)==0);
    x(i) = fp/neg_num;
    y(i) = tp/pos_num;
    auc = auc+(y(i)+y(i-1))*(x(i)-x(i-1))/2;    
end

x(m+1) = 1.0;
y(m+1) = 1.0;
% auc = auc + y(m)*x(m)/2;