function [ norm_data, max_val, min_val ] = normalize_me_python(data)
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
    max_val=max(max(max(data)));

    min_val=min(min(min(data)));
    if max_val~=min_val
        for i=1:size(data,1)
            for j=1:size(data,2)
                for k=1:size(data,3)
                    norm_data(i,j,k)=(data(i,j,k)-min_val)/(max_val-min_val);
                end
            end
        end
    else
        for i=1:size(data,1)
            for j=1:size(data,2)
                for k=1:size(data,3)
                    norm_data(i,j,k)=1;
                end
            end
        end
        min_val=min_val-0.000001;
    end

end

