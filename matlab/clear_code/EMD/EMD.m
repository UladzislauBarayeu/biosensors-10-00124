function [ epochsEMD,labels, number_of_IMFS ] = EMD( T, channel_names, type)
%END 
%   Detailed explanation goes here
%set number_of_IMFS
    number_of_IMFS=3;
    epochsEMD={};
    labels={};
    for Number_trial=1:size(T,2)     
        labeltrial={};
        trial={};
        
        for Number_chanel=1:size(T{Number_trial},2)
            signal=T{Number_trial}{Number_chanel};
            
            if type == 3
            %% MEMD
                IMFs=NAMEMD(signal);
                for Num_of_label=1:number_of_IMFS %?????
                    trial{Number_chanel}{Num_of_label}=IMFs(1,Num_of_label,:);
                    labeltrial{Number_chanel}{Num_of_label}=strcat(channel_names{Number_chanel}...
                        ,'_',num2str(Num_of_label));
                end
            else
                %% EMD
                if type == 2
                    IMFs=rParabEmd__L(signal,40,60,1);
                end
                if type == 1
                    IMFs=emd_lena(signal);
                end
                for Num_of_label=1:number_of_IMFS %?????
                    trial{Number_chanel}{Num_of_label}=IMFs(Num_of_label,:);
                    labeltrial{Number_chanel}{Num_of_label}=strcat(channel_names{Number_chanel}...
                        ,'_',num2str(Num_of_label));
                end
            end
            
        end
        epochsEMD{Number_trial}=trial;
    end
    labels=labeltrial;
end


