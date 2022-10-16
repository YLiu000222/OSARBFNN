clear all; close all; format long; format compact;clc
rng(3)
%% import data and plot
% X = load('F:\研究生\2022\temp\2spiral\2spiral.dat');
X = xlsread('F:\研究生\2022\LM RBF 修改\non.xlsx');
% inputs=X(:,1);
% outputs=X(:,2);
inputs = X(:, 1:2);
outputs = X(:, 3);%采用two spiral 一个是为1，一个是为-1的输出
% % separate into two categories for ploting purpose
% j1=0; j2=0;
[m,n] = size(inputs);%得到矩阵行和列m=194 n=2
% for i=1:m, %把X第三列的1和-1分开  分成X1和X2
%     if X(i,3)>0
%         j1=j1+1; X1(j1,1:2)=X(i,1:2); X1(j1,3)=X(i,3);
%     else
%         j2=j2+1; X2(j2,1:2)=X(i,1:2); X2(j2,3)=X(i,3);
%     end;
% end;
% initialize
actual_output_ = zeros(size(outputs));
centers = [];
weights_input = [];
weights_output = [1];
widths = [];
number_of_hidden_unit = 0;
h1=figure(1); clf; box on;
h5=figure(5); clf; box on;
%% random generation of centers and traditional solution at first
ni=1;
for kkk = 1:ni 
    number_of_hidden_unit = number_of_hidden_unit + 1;
    rr=sin(2*pi*(rand(1,2)-0.5)); %  generate randomly
    % rr=[-2.959595713163194   0.301758245008568];  % fixed, as an example
    centers = [centers; rr];  % location at maximum  32行
    weights_input = [weights_input; ones(1,n)];   % fixed values 里面的元素都是1  32行
    weights_output = [weights_output, 1];    % output weight =1 开始会生成1*ni 的为1的数字 后面会不断训练而改变
    widths = [widths, 2];                    % output width =2 开始会生成1*ni 的为2的数字，后面会不断训练而改变
    para_cur = parameter_combination(weights_output, widths, centers);
end
I = eye(length(para_cur));%121*121的单位矩阵
% other parameters
maximum_iteration = 100;
maximum_error = 0.001;
mu = 0.00001;%学习率
%% training process
[SSE(1)] = calculate_SSE(weights_input, weights_output,widths,centers,inputs,outputs);%计算出误差 每增加一个节点，计算其误差
fprintf('Number of RBF units = %d, iteration = 1, SSE = %6.10f\n',kkk,SSE(1));
for iter = 2:maximum_iteration
    jw = 0;
    [gradient, hessian] = calculate_gradient(weights_input, weights_output, widths, centers, inputs, outputs );%更新参数
    para_back = para_cur;%更新的
    while 1
        para_cur = para_back - (inv(hessian+mu*I)*gradient')';%公式3
        [weights_output, widths, centers] = parameter_divison(para_cur,number_of_hidden_unit,inputs);
        [SSE(iter)] = calculate_SSE(weights_input, weights_output,widths,centers,inputs,outputs);%用RBF计算误差
        if SSE(iter) <= SSE(iter-1)
            if mu > 10^-20;
                mu = mu/10;
            end;
            break;
        end;
        if mu < 10^20
            mu = mu*10;
        end;
        jw = jw + 1;
        if jw > 5
            break;
        end;
    end;    
    if mod(iter,10)==0; fprintf('Number of RBF units = %d, iteration = %d, SSE = %6.10f\n',kkk,iter, SSE(iter)); end
    ter(kkk)=SSE(iter);
    if abs(SSE(iter-1)-SSE(iter)) < maximum_error/1e8
        fprintf('Number of RBF units = %d, iteration = %d, SSE = %6.10f\n',kkk,iter, SSE(iter));
        break;
    end;
end;
error(kkk) = SSE(iter);
%% plot result
[actual_] = verification(weights_input, weights_output, widths, centers, inputs);
figure(h1);
nnn=101;
x_ =linspace(-6.5,6.5,nnn);
y_ =linspace(-6.5,6.5,nnn);
for i = 1:length(x_)
    for j = 1:length(y_)
        inputs_((i-1)*length(y_)+j,1) = x_(i);
        inputs_((i-1)*length(y_)+j,2) = y_(j);
    end;
end;
[actual_] = verification(weights_input, weights_output, widths, centers, inputs_);
reshaped=reshape(actual_, length(y_), length(x_));resha=reshaped./(1+abs(reshaped));
pcolor(x_,y_,resha); hold on; shading interp; colormap(bone);
contour(x_,y_,resha,1,'-g','LineWidth',2); hold on;
% scatter(X1(:,1),X1(:,2),'b.');scatter(X2(:,1),X2(:,2),'y.');
 scatter(centers(:,1),centers(:,2),'r*');scatter(centers(:,1),centers(:,2),'rh');
title([num2str(kkk),' RBF units']);hold off;

figure(h5);pcolor(x_,y_,resha); hold on;
shading interp; colormap(bone);
contour(x_,y_,resha,1,'-r','LineWidth',2); hold on;hold off;
%% number of RBF units increases from 1
[actual_output_] = verification(weights_input, weights_output, widths, centers, inputs);
over_times = zeros(30,1);          % 定义超过次数，此矩阵需大于over_centers
for kkk = ni+1:30 % visible problems for 30
    %% pruning 
    pruning = 1;
    if pruning == 1                    % 确认是否修剪
        over_centers = centers > 10 | centers < -10 | isnan(centers);  % 若中心值过大则直接删减 isnan 判断元素是否是nan，若是则返回1，不是0
        over_centers = over_centers(:,1) | over_centers(:,2);          % 合并两列的值
        over_times(1:size(over_centers,1)) = over_times(1:size(over_centers,1)) + over_centers;  % 记录超过次数 size(over_centers,1)返回over_centers的行数
        del_num = find(over_times>5);
        centers(del_num,:) = [];  % 
        weights_input(del_num,:) = [];
        weights_output(del_num+1) = [];
        widths(del_num) = [];
        number_of_hidden_unit = number_of_hidden_unit - size(del_num,1);
    end
    [maxi_, index_1] = max(abs(outputs-actual_output_));   % find maximum 找到最大误差点 
    number_of_hidden_unit = number_of_hidden_unit + 1;     % 隐藏层个数
    centers = [centers; inputs(index_1,:)];           % location at maximum   中心初始化为最大误差
    weights_input = [weights_input; ones(1,n)];   % fixed values          输入权重缩放初始化
    weights_output = [weights_output, 1];             % output weight = 1     输出权重缩放初始化
    widths = [widths,0.1];                            % output width = 1      输出宽度初始化
    para_cur = parameter_combination(weights_output, widths, centers);
    I = eye(length(para_cur));
    % other parameters 
    maximum_iteration = 100;        % 最大迭代次数
    mu = 0.1;                       % 学习率
    % training process              
    [SSE(1)] = calculate_SSE(weights_input, weights_output,widths,centers,inputs,outputs);
    fprintf('Number of RBF units = %d, iteration = 1, SSE = %6.10f\n',kkk,SSE(1));
    for iter = 2:maximum_iteration
        jw = 0;
        [gradient, hessian] = calculate_gradient(weights_input, weights_output, widths, centers, inputs, outputs );
        para_back = para_cur;
        while 1
            para_cur = para_back - (inv(hessian+mu*I)*gradient')';
            [weights_output, widths, centers] = parameter_divison(para_cur,number_of_hidden_unit,inputs);
            [SSE(iter)] = calculate_SSE(weights_input, weights_output,widths,centers,inputs,outputs);
            if SSE(iter) <= SSE(iter-1)
                if mu > 10^-20; mu = mu/10; end
                break;
            end
            if mu < 10^20; mu = mu*10; end
            jw = jw + 1;
            if jw > 5; break; end
        end
        if mod(iter,10)==0,   fprintf('Number of RBF units = %d, iteration = %d, SSE = %6.10f\n',kkk,iter, SSE(iter)); end
        ter(kkk)=SSE(iter);
        if abs(SSE(iter-1)-SSE(iter)) < maximum_error/1e8
            fprintf('Number of RBF units = %d, iteration = %d, SSE = %6.10f\n',kkk,iter, SSE(iter));
            break;
        end
    end
    error(kkk) = SSE(iter);
    [actual_] = verification(weights_input, weights_output, widths, centers, inputs);

    %% plot results
    figure(h1);
    nnn=101;
    x_ =linspace(-6.5,6.5,nnn);y_ =linspace(-6.5,6.5,nnn);
    for i = 1:length(x_)
        for j = 1:length(y_)
            inputs_((i-1)*length(y_)+j,1) = x_(i);
            inputs_((i-1)*length(y_)+j,2) = y_(j);
        end
    end
    [actual_] = verification(weights_input, weights_output, widths, centers, inputs_);    
    reshaped=reshape(actual_, length(y_), length(x_));
    resha=reshaped./(1+abs(reshaped));
    pcolor(x_,y_,resha); hold on;
    shading interp; colormap(bone);
    contour(x_,y_,resha,1,'-g','LineWidth',2); hold on;
    % scatter(X1(:,1),X1(:,2),'b.');scatter(X2(:,1),X2(:,2),'y.');
    scatter(centers(:,1),centers(:,2),'r*');scatter(centers(:,1),centers(:,2),'rh');
    title([num2str(kkk),' RBF units']);hold off;
    
    figure(h5); pcolor(x_,y_,resha); hold on;
    shading interp; colormap(bone);
    contour(x_,y_,resha,1,'-r','LineWidth',2); hold off;
    
    if SSE(iter) < maximum_error
        fprintf('Number of RBF units = %d, iteration = %d, SSE = %6.10f\n',kkk,iter, SSE(iter));
        break;
    end
    [actual_output_] = verification(weights_input, weights_output, widths, centers, inputs);



end

figure(2); clf; semilogy(ter);xlabel('RBF unit(s)');ylabel('SSE');grid on; box on;
figure(3); clf;resha=reshaped./(1+abs(reshaped));pcolor(x_,y_,resha); hold on;
shading interp; colormap(bone);contour(x_,y_,resha,1,'-r','LineWidth',2); hold on;
% scatter(X1(:,1),X1(:,2),'b.');scatter(X2(:,1),X2(:,2),'y.');title('final result');hold off;