function [ features, freq_labels ] = freq_feat( T, labels, sample_rate )

    for Number_trial=1:size(T,2)
        trial=[];
        
        for Number_chanel=1:size(T{Number_trial},2)
            signal=T{Number_trial}{Number_chanel};
            
            param.Fs=sample_rate;
            param.tapers=[3 5];
            param.fpass=[0.5 42];
           
            [~,~,~,S1,~,f]=coherencyc(signal,signal,param);
            
            %% log scale
%             log_scale = logspace(0,1.6,13);
%             PSDs={};
%             for j=1:size(log_scale,2)-1
%                 PSDs{j}=[];
%             end
%             for i=1:size(f,2)
%                 for j=1:size(log_scale,2)-1
%                     if f(i)>log_scale(j) && f(i)<log_scale(j+1)
%                         PSDs{j}(end+1)=S1(i);
%                     end
%                 end
%             end
%             for j=1:size(PSDs,2)
%                 trial(Number_chanel, j)=mean(PSDs{j});
%                 freq_labels{Number_chanel, j}=strcat(labels{Number_chanel}...
%                 ,'_from_',string(log_scale(j)),'Hz_to_',string(log_scale(j+1)),'Hz');
%             end
            %% mu betha
            PSDs={[],[]};
            for i=1:size(f,2)
                if f(i)>7.5 && f(i)<12.5
                    PSDs{1}(end+1)=S1(i);
                end
                if f(i)>16 && f(i)<31
                    PSDs{2}(end+1)=S1(i);
                end
            end
            trial(Number_chanel, 1)=mean(PSDs{1});
            freq_labels{Number_chanel, 1}=strcat(labels{Number_chanel}...
                ,'_mu');
            
            trial(Number_chanel, 2)=mean(PSDs{2});
            freq_labels{Number_chanel, 2}=strcat(labels{Number_chanel}...
                ,'_betta');
        end
        features{Number_trial}=trial;
    end
    
end

