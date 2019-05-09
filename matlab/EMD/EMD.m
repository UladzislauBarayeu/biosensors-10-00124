function [ epochsEMD,labels, number_of_IMFS ] = EMD( T, channel_names)
%END 
%   Detailed explanation goes here
%set number_of_IMFS
    number_of_IMFS=5;
    epochsEMD={};
    labels={};
    for Number_trial=1:size(T,2)
        labeltrial={};
        trial={};
        
        for Number_chanel=1:size(T{Number_trial},2)
            signal=T{Number_trial}{Number_chanel};
            
            IMFs=emd_lena(signal);
            %for i=1:size(IMFs,1)
            %    figure;
            %    plot(IMFs(i,:));
            %end
            
            %IMFs=rParabEmd__L(signal,40,60,1);
            %for i=1:size(IMFs,1)
            %    figure;
            %    plot(IMFs(:,i));
            %end
            for Num_of_label=1:number_of_IMFS %?????
                trial{Number_chanel}{Num_of_label}=IMFs(Num_of_label,:);
                labeltrial{Number_chanel}{Num_of_label}=strcat(channel_names{Number_chanel}...
                    ,'_',num2str(Num_of_label));
            end
        end
        epochsEMD{Number_trial}=trial;
    end
    labels=labeltrial;
end


