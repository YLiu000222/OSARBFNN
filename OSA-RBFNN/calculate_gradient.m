function [gradient, hessian] = calculate_gradient(ww, weights, widths, centers, inputs, outputs)
%calculate_gradient                        weights_input, weights_output, widths, centers, inputs, outputs 
%   gradient calculation
[p1,p2] = size(weights);%输出权重
[p3,p4] = size(centers);
[p5,p6] = size(widths);
%初始化
gradient = zeros(1,p1*p2+p3*p4+p5*p6);
hessian = zeros(p1*p2+p3*p4+p5*p6,p1*p2+p3*p4+p5*p6);

[m,n] = size(inputs);
for i = 1:m
    net = weights(1);
    for j = 1:p3
        node(j) = exp(-sum((ww(j,:).*inputs(i,:)-centers(j,:)).^2)/widths(j));%公式1
        net = net + node(j)*weights(j+1);
    end;
    % for g_weight
    out = net;
    de = 1;
    err = outputs(i,1) - out;
    J_weight(1) = -de;
    for j = 2:p2
        J_weight(j) = J_weight(1)*node(j-1);
    end;
    % for g_center
    for j = 1:p3
        J_center(j,:) = (-1)*weights(j+1)*node(j)*2*(ww(j,:).*inputs(i,:)-centers(j,:))./widths(j);
        J_width(j) = (-1)*weights(j+1)*node(j)*sum((ww(j,:).*inputs(i,:)-centers(j,:)).^2)/widths(j)^2;
        for k = 1:n
            J_ww(j,k) = (-1)*weights(j+1)*node(j)*(-1)/widths(j)*2*(ww(j,k)*inputs(i,k)-centers(j,k))*inputs(i,k);
        end;
    end;
    J = parameter_combination(J_weight, J_width, J_center);
    gradient = gradient + err*J;
    hessian = hessian + J'*J;
end;

