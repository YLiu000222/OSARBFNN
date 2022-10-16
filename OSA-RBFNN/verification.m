function [output] = verification(weights_input, weights_output, widths, centers, inputs)
%verification:                  weights_input, weights_output, widths, centers, inputs
%   Verification process
[m,n] = size(inputs);
[p,q] = size(centers);
for i = 1:m
    count = weights_output(1);
    for j = 1:p
        count = count + weights_output(j+1)*exp(-sum((weights_input(j,:).*inputs(i,:)-centers(j,:)).^2)/widths(j));%公式1和公式2
    end;
    output(i,1) = count; %输出权重
end;

