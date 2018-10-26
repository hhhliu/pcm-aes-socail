tf=load('/home/liuhuihui/ME/data/tag_feature.mat');
A=tf.tag_feature;
disp(size(A));
opt=statset('maxiter',100,'display','final');
[W,H,D]=nnmf(A,200,'rep',20,'opt',opt,'alg','mult');
%disp(W*H);
sumA=sum(sum(W*H));
disp(sumA);
disp(D);

save('/home/liuhuihui/ME/data/tag_feature_nnmf.mat',W*H);
