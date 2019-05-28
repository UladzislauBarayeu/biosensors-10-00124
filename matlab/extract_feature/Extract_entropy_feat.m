function [ result_feature, entropylabels ] = Extract_entropy_feat( T,labels)
%EXTRACT_ENTROPY_FEAT Summary of this function goes here
%   Detailed explanation goes here
    entropylabels={};
    for Number_of_trial=1:size(T,2)
        trial=[];
        for Number_of_channels=1:size(T{Number_of_trial},2)
            iterator=1;
            for Number_of_IMF=1:size(T{Number_of_trial}{Number_of_channels},2)
                data=T{Number_of_trial}{Number_of_channels}{Number_of_IMF};
                
                %shannon
                ShanonEntrop=wentropy(data,'shannon');
                trial(Number_of_channels, iterator)=ShanonEntrop;
                entropylabels{Number_of_channels, iterator}=strcat(labels{Number_of_channels}...
                    {Number_of_IMF},'_shannon');
                iterator=iterator+1;
                
                %log energy
                log_energy=wentropy(data,'log energy');
                trial(Number_of_channels, iterator)=log_energy;
                entropylabels{Number_of_channels, iterator}=strcat(labels{Number_of_channels}...
                    {Number_of_IMF},'_log_energy');
                iterator=iterator+1;
                
                %sample and Ap en
                sd = std(data);
                r = 0.15*sd;
                M = 2;
                %tau = 1;
                
                SampelEntropy=sampen(M, r, data);
                trial(Number_of_channels, iterator)=SampelEntropy;
                entropylabels{Number_of_channels,iterator}=strcat(labels{Number_of_channels}...
                    {Number_of_IMF},'_SampEn');
                iterator=iterator+1;
                
                ApEntropy=ApEn(data, M, r);
                trial(Number_of_channels, iterator)=ApEntropy;
                entropylabels{Number_of_channels, iterator}=strcat(labels{Number_of_channels}...
                    {Number_of_IMF},'_ApEn');
                iterator=iterator+1;
            end
        end
        result_feature{Number_of_trial}=trial;
    end
    

end

