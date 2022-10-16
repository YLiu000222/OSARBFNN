function [SSE] = calculate_SSE(ww, weights,widths,centers,inputs,outputs)
%calculate_SSE                weights_input, weights_output,widths,centers,inputs,outputs
%   Error(Sum Square Error) calculation.
[m,n] = size(inputs);%m=194 n=2
[p,q] = size(centers);%p=30 n=2
SSE = 0;
for i = 1:m
    count = weights(1);
    for j = 1:p
        count = count + weights(j+1)*exp(-sum((ww(j,:).*inputs(i,:)-centers(j,:)).^2)/widths(j));%公式（1）和（2）计算RBF output
    end;
    SSE = SSE + (count - outputs(i,1))^2;%公式6 计算经过RBF的和所预先设置输出的 之间的误差
end;
SSE = SSE/m;%公式12 mse 均方误差
end

