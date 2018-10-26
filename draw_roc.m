% load('/home/liuhuihui/ME/pagerank_result/roc/ava/xx.mat','xx');
% load('/home/liuhuihui/ME/pagerank_result/roc/ava/yy.mat','yy');
% load('/home/liuhuihui/ME/pagerank_result/roc/ava/aucs.mat','aucs');


load('/home/liuhuihui/ME/pagerank_result/dims/ava/xx.mat','xx');
load('/home/liuhuihui/ME/pagerank_result/dims/ava/yy.mat','yy');
load('/home/liuhuihui/ME/pagerank_result/dims/ava/aucs.mat','aucs');

% plot(xx(:,1),yy(:,1));
% hold on
% plot(xx(:,2),yy(:,2));
% 
% t = sprintf('auc = %0.3f',auc);
% legend(t,'FontName','FontSize',15,'Location','Southeast');
% set(gca,'FontSize',15);
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',1);
% % text(0.7,0.1,['auc =',t]);

figure('Color',[1 1 1])
%plot(xx(:,1),yy(:,1),'-ro',xx(:,2),yy(:,2),'-*g',xx(:,3),yy(:,3),'-s',xx(:,4),yy(:,4),'-m+',xx(:,5),yy(:,5),'-d',xx(:,6),yy(:,6),'-b','LineWidth',1, 'MarkerSize',5);
% plot(xx(:,1),yy(:,1),xx(:,2),yy(:,2),xx(:,3),yy(:,3),xx(:,4),yy(:,4),'m',xx(:,5),yy(:,5),'b',xx(:,6),yy(:,6),xx(:,7),yy(:,7),xx(:,8),yy(:,8),'r',xx(:,9),yy(:,9),'LineWidth',2, 'MarkerSize',5);
% plot(xx(:,1),yy(:,1),xx(:,2),yy(:,2),xx(:,3),yy(:,3),xx(:,4),yy(:,4),'LineWidth',2, 'MarkerSize',5);
% plot(xx(:,1),yy(:,1),'LineWidth',2, 'MarkerSize',5);
plot(xx(:,1),yy(:,1),'y',xx(:,2),yy(:,2),'m',xx(:,3),yy(:,3),'c',xx(:,4),yy(:,4),'r',xx(:,5),yy(:,5),'g',xx(:,6),yy(:,6),'b',xx(:,8),yy(:,8),'g','LineWidth',1, 'MarkerSize',5);

% legend('lu','tag','user','group','tag+user','user+group','tag_group','SIAA','resnet50');
legend('32','64','128','256','512','1024','lu');
xlabel('False Positive Rate','fontsize',14);
ylabel('True Positive Rate','fontsize',14);
set(gca,'FontSize',14);
% title('AVA','fontsize',14,'fontweight','b');
