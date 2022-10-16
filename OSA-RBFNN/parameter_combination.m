function [vector] = parameter_combination(weights_output, widths, centers)
% parameter_combination:
%   combination of parameters  
 [p3,p4] = size(centers);
vector = [weights_output widths reshape(centers',1,p3*p4)];%reshape 按列读取 最终生成一行60列的（centers）
end% vector=31+30+60=121

