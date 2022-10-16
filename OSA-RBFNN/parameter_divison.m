function [weights_output, widths, centers] = parameter_divison(vector, num, data)
% parameter_divison:                                          para_cur,number_of_hidden_unit,inputs
%   Divide parameters.
[row, col] = size(data);%194*2
for i = 1:(num+1)
    weights_output(1,i) = vector(1,i);
end;
for i = 1:num
    widths(1,i) = vector(1, num+1+i);
end;
for i = 1:num
    for j = 1:col
        centers(i,j) = vector(1,2*num+1+(i-1)*col+j);
    end;
end;

