function [ epochsEMD,labels ] = EMD( T, channel_names, type, number_of_IMFS)
%END 
%   Detailed explanation goes here
%set number_of_IMFS
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
                if type == 0
                    IMFs=emd(signal);
                end
                if type == 2
                    IMFs=rParabEmd__L(signal,40,60,1);
                end
                if type == 1
                    IMFs=emd_lena(signal);
                    
                end
                for Num_of_label=1:number_of_IMFS 
                    trial{Number_chanel}{Num_of_label}=IMFs(Num_of_label,:);
                    labeltrial{Number_chanel}{Num_of_label}=strcat(channel_names{Number_chanel}...
                        ,'_',num2str(Num_of_label));
                    %% test IMF power
%                     param.Fs=160;
%                     param.tapers=[3 5];
%                     param.fpass=[0.5 42];
%                     [~,~,~,S1,~,f]=coherencyc(IMFs(Num_of_label,:),IMFs(Num_of_label,:),param);
%                     plot(f,S1);
%                     title(strcat('spectrum IMF-',num2str(Num_of_label)))
%                     xlabel('freq')
%                     ylabel('PSD')
%                     outputjpgDir = strcat('figures/IMF/');
%                     if ~exist(outputjpgDir, 'dir')
%                             mkdir(outputjpgDir);
%                     end
%                     saveas(gcf,strcat(outputjpgDir,num2str(Num_of_label),'.jpg'));

                end
            end
            
        end
        epochsEMD{Number_trial}=trial;
    end
    labels=labeltrial;
end


