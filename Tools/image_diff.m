function [result] = image_diff(offset, ori, rec)
% result = sqrt(sum(sum( (ori - (rec + offset)).^2 ) ));
% result = (ori - (rec + offset));
result = angle(rec *exp(-1i*offset)) - angle(ori);
end