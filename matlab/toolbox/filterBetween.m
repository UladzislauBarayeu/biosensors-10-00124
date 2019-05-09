function filteredRes = filterBetween(EEG, sampleRate, low, high, order) 


    low_rad=(2/sampleRate)*low;
    high_rad=(2/sampleRate)*high;
    %open quation about order
    filt=fir1(order, [low_rad high_rad]);
    group_delay = median(grpdelay(filt));
    filteredEEG=filter(filt,1,EEG);
    
    

    result_signal(1:((size(EEG,1)-group_delay+1)))=filteredEEG(group_delay:end);
    band_point=filteredEEG(end);
    count=0;
    for i=size(filteredEEG,1):-1:2
        if count>=3
            break
        end
        if (filteredEEG(i)>=band_point && filteredEEG(i-1)<=band_point)||...
                (filteredEEG(i)<=band_point && filteredEEG(i-1)>=band_point)
            count=count+1;
        end
    end
    end_signal=filteredEEG((i+1):end);
    while size(result_signal,2)<size(EEG,1)
        result_signal((end+1):(end+size(end_signal,1)))=end_signal;
    end
    filteredRes=result_signal(1:size(EEG,1));
    
    %plot( EEG)
    %hold on;
    %plot(filteredRes)
    %hold off;
    
end