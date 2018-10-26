% xx = [];
% yy = [];
% aucs = [];

% load('/home/liuhuihui/ME/pagerank_result/aes_Lu/cnn-aes-Lu-ava.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/Lu+tag/tag2_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/Lu+user/user2_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/Lu+group/group2_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/Lu+tag+user/tag_user2_ava_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/Lu+user+group/user_group2_ava_Lu.mat');
% predict = scores;
% 
% load('/home/liuhuihui/ME/pagerank_result/Lu+tag+group/tag_group2_avatest_Lu.mat');
% predict = scores;

load('/home/liuhuihui/ME/pagerank_result/SIAA=Lu+tag+user+group/method2_avatest_Lu.mat');
predict = scores;

load('/home/liuhuihui/baseline/ava/label_test.mat');
real = label_test;
[x,y,auc] = rocauc(predict,real);
xx = [xx,x];
yy = [yy,y];
aucs = [aucs,auc];

save('/home/liuhuihui/ME/pagerank_result/roc/ava/xx.mat','xx');
save('/home/liuhuihui/ME/pagerank_result/roc/ava/yy.mat','yy');
save('/home/liuhuihui/ME/pagerank_result/roc/ava/aucs.mat','aucs');
