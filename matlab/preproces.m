function [] = preproces( task, folderpath )
%% preproces data for task number T

size_of_subjects=108;

iterator_sub=0;
for subject=1:size_of_subjects
    if subject==88 || subject==92 || subject==100 
        continue
    end
    iterator_sub=iterator_sub+1;
    filepath=strcat(folderpath,num2str(subject.','S%03d'),'/',num2str(subject.','S%03d'),'R');
    
    switch task
        case 1
            filename1=strcat(filepath,'03.edf');
            filename2=strcat(filepath,'07.edf');
            filename3=strcat(filepath,'11.edf');
        case 2
            filename1=strcat(filepath,'04.edf');
            filename2=strcat(filepath,'08.edf');
            filename3=strcat(filepath,'12.edf');
        case 3
            filename1=strcat(filepath,'05.edf');
            filename2=strcat(filepath,'09.edf');
            filename3=strcat(filepath,'13.edf');
        case 4
            filename1=strcat(filepath,'06.edf');
            filename2=strcat(filepath,'10.edf');
            filename3=strcat(filepath,'14.edf');
    end

    [data1, annotation1]=readEDF(filename1);
    [data2, annotation2]=readEDF(filename2);
    [data3, annotation3]=readEDF(filename3);

    T1={};T2={};
    %% combine data
    % data 1
    for i=2:2:size(annotation1.annotation.event,2)
        data={};
        for nb_channel=1:size(data1,2)
            start_val=int32(annotation1.annotation.starttime(i-1)*160);
            if start_val==0
                start_val=1;
            end
            end_val=int32((annotation1.annotation.starttime(i-1)+annotation1.annotation.duration(i-1))*160);
            %baseline2{nb_channel}=wconv1(data1{nb_channel}(start_val:end_val,1),ones(300,1),'same')/300;
            baseline{nb_channel}=mean(data1{nb_channel}(start_val:end_val,1));
            %data2=data1{nb_channel}(start_val:end_val,1);
            %plot(data2); hold on;
            %plot(baseline2{nb_channel}); hold off;
            %plot(data2-baseline2{nb_channel});
        end

        for nb_channel=1:size(data1,2)
            start_val=int32(annotation1.annotation.starttime(i)*160);
            if start_val==0
                start_val=1;
            end
            end_val=int32((annotation1.annotation.starttime(i)+annotation1.annotation.duration(i))*160);
            data{nb_channel}=filterBetween((data1{nb_channel}(start_val:end_val,1)-baseline{nb_channel}),...
                annotation1.samplerate(nb_channel),1,50,40);
        end

        T_number=annotation1.annotation.event{i};
        if T_number=='T1'
            T1{end+1}=data;
        else
            T2{end+1}=data;
        end
    end

    %data 2
    for i=2:2:size(annotation2.annotation.event,2)
        data={};
        for nb_channel=1:size(data2,2)
            start_val=int32(annotation2.annotation.starttime(i-1)*160);
            if start_val==0
                start_val=1;
            end
            end_val=int32((annotation2.annotation.starttime(i-1)+annotation2.annotation.duration(i-1))*160);
            baseline{nb_channel}=mean(data2{nb_channel}(start_val:end_val,1));
        end

        for nb_channel=1:size(data2,2)
            start_val=int32(annotation2.annotation.starttime(i)*160);
            if start_val==0
                start_val=1;
            end
            end_val=int32((annotation2.annotation.starttime(i)+annotation2.annotation.duration(i))*160);
            data{nb_channel}=filterBetween((data2{nb_channel}(start_val:end_val,1)-baseline{nb_channel}),...
                annotation2.samplerate(nb_channel),1,50,40);
        end

        T_number=annotation2.annotation.event{i};
        if T_number=='T1'
            T1{end+1}=data;
        else
            T2{end+1}=data;
        end
    end


    % data 3
    for i=2:2:size(annotation3.annotation.event,2)
        data={};
        for nb_channel=1:size(data3,2)
            start_val=int32(annotation3.annotation.starttime(i-1)*160);
            if start_val==0
                start_val=1;
            end
            end_val=int32((annotation3.annotation.starttime(i-1)+annotation3.annotation.duration(i-1))*160);
            baseline{nb_channel}=mean(data3{nb_channel}(start_val:end_val,1));
        end

        for nb_channel=1:size(data3,2)
            start_val=int32(annotation3.annotation.starttime(i)*160);
            if start_val==0
                start_val=1;
            end
            end_val=int32((annotation3.annotation.starttime(i)+annotation3.annotation.duration(i))*160);
            data{nb_channel}=filterBetween((data3{nb_channel}(start_val:end_val,1)-baseline{nb_channel}),...
                annotation3.samplerate(nb_channel),1,50,40);
        end

        T_number=annotation3.annotation.event{i};
        if T_number=='T1'
            T1{end+1}=data;
        else
            T2{end+1}=data;
        end
    end

    %% Cut Data
    size_of_window=320;%2 sec sample_rate=160
    step=80;% overlap 25%
    %T1
    for channel=1:length(T1{1})
        trial=1;
        for trial_old=1:length(T1)
            signal=T1{trial_old}{channel};
            k=0;
            while length(signal)-size_of_window-step*k>0
                T1_new{trial}{channel}=signal((1+step*k):(size_of_window+step*k));
                k=k+1;
                trial=trial+1;
            end
        end
    end
    %T2
    for channel=1:length(T2{1})
        trial=1;
        for trial_old=1:length(T2)
            signal=T2{trial_old}{channel};
            k=0;
            while length(signal)-size_of_window-step*k>0
                T2_new{trial}{channel}=signal((1+step*k):(size_of_window+step*k));
                k=k+1;
                trial=trial+1;
            end
        end
    end
    %%
    clear T2 T1
    T2=T2_new; T1=T1_new;
    clear T2_new T1_new
    %%  Save
    channel_names=annotation1.labels;
    sample_rate=annotation1.samplerate(1);
    % Define the folder where to store the data
    outputDir = sprintf(strcat('Data/preproces/task',num2str(task),'/'));
    % Check if the folder exists, and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    % Define the filename to store the data
    outputfilename = strcat( outputDir, num2str(iterator_sub),'.mat');
    % Write it to disk  
    save(outputfilename, 'T1', 'T2','channel_names','sample_rate');
    
    
    clearvars -except size_of_subjects task subject folderpath iterator_sub
end


end

