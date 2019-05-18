function IMFs = NAMEMD(EEG)
%% By Arno Libert 16-5-19
% Will return IMFs for EEG signals, max nr of channels is 16
% input: EEG: of the form mxn with m being the channels and n being the
% sampels
% output: IMFs

    if size(EEG,1) == 1
        EEG(2,:)=wgn(size(EEG,2),1,0);
        EEG(3,:)=wgn(size(EEG,2),1,0);
    else
        for i = 1:floor(size(EEG,1)*2)
            EEG(size(EEG,1)+i,:)=wgn(size(EEG,2),1,0); 
        end
    end

    IMFs = memd(EEG);
                
end