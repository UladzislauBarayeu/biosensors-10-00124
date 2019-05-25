function [ features, freq_labels ] = freq_feat( T, labels, sample_rate )
%FREQ_FEAT Summary of this function goes here
%   Detailed explanation goes here

    for Number_trial=1:size(T,2)
        trial=[];
        
        for Number_chanel=1:size(T{Number_trial},2)
            signal=T{Number_trial}{Number_chanel};
            
            param.Fs=sample_rate;
            param.tapers=[3 5];
            param.fpass=[0.5 42];
           
            [~,~,~,S1,~,f]=coherencyc(signal,signal,param);
            
            % 1 ?? 4 
            % 4 ?? 8 
            % 8 ?? 13
            % 13 ?? 40
            alphas=[];bettas=[];thetas=[];deltas=[];
            log_scale = logspace(0,1.6,13);
            PSDs={};
            for j=1:size(log_scale,2)-1
                PSDs{j}=[];
            end
            for i=1:size(f,2)
                for j=1:size(log_scale,2)-1
                    if f(i)>log_scale(j) && f(i)<log_scale(j+1)
                        PSDs{j}(end+1)=S1(i);
                    end
                end
%                 if(f(i)>1 && f(i)<4)
%                     deltas(end+1)=S1(i);
%                 end
%                 if(f(i)>4 && f(i)<8)
%                     thetas(end+1)=S1(i);
%                 end
%                 if(f(i)>8 && f(i)<13)
%                     alphas(end+1)=S1(i);
%                 end
%                 if(f(i)>13 && f(i)<40)
%                     bettas(end+1)=S1(i);
%                 end
            end
            for j=1:size(log_scale,2)-1
                trial(Number_chanel, j)=mean(PSDs{j});
                freq_labels{Number_chanel, j}=strcat(labels{Number_chanel}...
                ,'_from_',string(log_scale(j)),'Hz_to_',string(log_scale(j+1)),'Hz');
            end
            
%             trial(Number_chanel, 1)=mean(deltas);
%             freq_labels{Number_chanel, 1}=strcat(labels{Number_chanel}...
%                 ,'_delta');
%             
%             trial(Number_chanel, 2)=mean(thetas);
%             freq_labels{Number_chanel, 2}=strcat(labels{Number_chanel}...
%                 ,'_theta');
%             
%             trial(Number_chanel, 3)=mean(alphas);
%             freq_labels{Number_chanel, 3}=strcat(labels{Number_chanel}...
%                 ,'_alpha');
%             
%             trial(Number_chanel, 4)=mean(bettas);
%             freq_labels{Number_chanel, 4}=strcat(labels{Number_chanel}...
%                 ,'_betta');
        end
        features{Number_trial}=trial;
    end
    
end

