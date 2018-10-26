% xx = [];
% yy = [];
% aucs = [];
% 
% load('/home/liuhuihui/ME/pagerank_result/dims/method2_32_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/dims/method2_64_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/dims/method2_128_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/SIAA=Lu+tag+user+group/method2_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/dims/method2_512_avatest_Lu.mat');
% predict = scores;


% load('/home/liuhuihui/ME/pagerank_result/dims/method2_1024_avatest_Lu.mat');
% predict = scores;

% load('/home/liuhuihui/ME/pagerank_result/resnet_aes_ava.mat');
% predict = scores;

load('/home/liuhuihui/ME/pagerank_result/aes_Lu/cnn-aes-Lu-ava.mat');
predict = scores;

load('/home/liuhuihui/baseline/ava/label_test.mat');
real = label_test;
[x,y,auc] = rocauc(predict,real);
xx = [xx,x];
yy = [yy,y];
aucs = [aucs,auc];

save('/home/liuhuihui/ME/pagerank_result/dims/ava/xx.mat','xx');
save('/home/liuhuihui/ME/pagerank_result/dims/ava/yy.mat','yy');
save('/home/liuhuihui/ME/pagerank_result/dims/ava/aucs.mat','aucs');
