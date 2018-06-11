load ('010153-81-010321-010454-010523-37-69-010648.mat')
temp = size(P_Train,1);
sum1 = 0;
sum2 = 0;
%ѵ����1655��
P_train = P_Train(1:3181,:)';
T_train = T_Train(1:3181,:)';

%���Լ�189��
P_test = P_Test(1:477,:)';
T_test = T_Test(1:477,:)';
N = size(P_test,2);
%���ݹ�һ��
[P_train1,inputps] = mapminmax(P_train);
[T_train1,outputps] = mapminmax(T_train);
%��������
net = newff(P_train,T_train,6);
%����ѵ������
net.trainParam.epochs = 1000;
net.trainParam.goal = 0.001;
net.trainParam.lr = 0.01;
%ѵ������
net = train(net,P_train,T_train);
%���Բ��������ݹ�һ��
%P_test1 = mapminmax('apply',P_test,inputps);
y = sim(net,P_test);
%�����������һ��
BP_output = mapminmax('reverse',y,outputps);
err = mse(net,T_test,BP_output)
for i=1:length(BP_output)
    sum1=sum1+((BP_output(i)-T_test(i)).^2);
 
end
BP_RMSE = sqrt(sum1/length(BP_output))

result = [T_test' BP_output']
%figure;
%plot(1:N,T_test(1,:),'b-*',1:N,BP_output(1,:),'r:p');
%legend('��ʵ����','Ԥ�����');
%xlabel('����������');
%ylabel('����ֵ');
%figure;
%plot(1:N,T_test(2,:),'b-*',1:N,BP_output(2,:),'r:p');
%legend('��ʵ����','Ԥ�ⷽ��');
%xlabel('����������');
%ylabel('����ֵ');
%����ѵ���õ�����
save data net inputps outputps




