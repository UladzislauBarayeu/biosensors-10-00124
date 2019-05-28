function [ norm_array, max_val, min_val] = normalize_me( array )
%NORMALIZE_ME Summary of this function goes here
%   Detailed explanation goes here
    max_val=max(array);
    min_val=min(array);
    if max_val~=min_val
        for iterator=1:size(array,2)
            norm_array(iterator)=(array(iterator)-min_val)/(max_val-min_val);
        end
    else
        for iterator=1:size(array,2)
            norm_array(iterator)=1;
        end
        min_val=min_val-0.000001;
    end
    

end

