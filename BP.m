load ('010153-81-010321-010454-010523-37-69-010648.mat')
temp = size(P_Train,1);
sum1 = 0;
sum2 = 0;
%训练集1655个
P_train = P_Train(1:3181,:)';
T_train = T_Train(1:3181,:)';

%测试集189个
P_test = P_Test(1:477,:)';
T_test = T_Test(1:477,:)';
N = size(P_test,2);
%数据归一化
[P_train1,inputps] = mapminmax(P_train);
[T_train1,outputps] = mapminmax(T_train);
%创建网络
net = newff(P_train,T_train,6);
%设置训练参数
net.trainParam.epochs = 1000;
net.trainParam.goal = 0.001;
net.trainParam.lr = 0.01;
%训练网络
net = train(net,P_train,T_train);
%测试并测试数据归一化
%P_test1 = mapminmax('apply',P_test,inputps);
y = sim(net,P_test);
%网络输出反归一化
BP_output = mapminmax('reverse',y,outputps);
err = mse(net,T_test,BP_output)
for i=1:length(BP_output)
    sum1=sum1+((BP_output(i)-T_test(i)).^2);
 
end
BP_RMSE = sqrt(sum1/length(BP_output))

result = [T_test' BP_output']
%figure;
%plot(1:N,T_test(1,:),'b-*',1:N,BP_output(1,:),'r:p');
%legend('真实距离','预测距离');
%xlabel('测试样本数');
%ylabel('距离值');
%figure;
%plot(1:N,T_test(2,:),'b-*',1:N,BP_output(2,:),'r:p');
%legend('真实方向','预测方向');
%xlabel('测试样本数');
%ylabel('方向值');
%保存训练好的网络
save data net inputps outputps




