clear all
load('Data/cross/cross')
for Number_feat=1:size(R,1)
    for ch1=1:size(R,2)
        for ch2=1:size(R,3)
            R_feat(ch1,ch2)=R(Number_feat,ch1,ch2);
        end
    end

    y=[1 64];x=[1 64];
    imagesc(x,y, R_feat, [0 1]);

    %title('cross-correlation all channel');
    set(gca,'YDir','normal')
    colormap(gca,jet);
    c=colorbar;
    c.Label.String = 'R';
    %save file
    outputjpgDir = strcat('figures/cross/');
    if ~exist(outputjpgDir, 'dir')
            mkdir(outputjpgDir);
    end
    namefile=strcat('%s/%s',num2str(Number_feat),'.jpg');
    outputjpgname = sprintf(namefile, outputjpgDir, 'cross-correlation');
    saveas(gcf,outputjpgname);
    hold off;
    
end

y=[1 64];x=[1 64];
imagesc(x,y, Result_R, [0 1]);

%title('cross-correlation all channel');
set(gca,'YDir','normal')
xlabel('Channels')
ylabel('Channels')
colormap(gca,jet);
c=colorbar;
c.Label.String = 'R';
%save file
outputjpgDir = strcat('figures/cross/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s/%s','mean R','.jpg');
outputjpgname = sprintf(namefile, outputjpgDir, 'cross-correlation');
saveas(gcf,outputjpgname);
hold off;

%STD
y=[1 64];x=[1 64];
imagesc(x,y, Std_R, [0 1]);

%title('cross-correlation all channel');
xlabel('Channels')
ylabel('Channels')
set(gca,'YDir','normal')
colormap(gca,jet);
c=colorbar;
c.Label.String = 'R';
%save file
outputjpgDir = strcat('figures/cross/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s/%s','STD','.jpg');
outputjpgname = sprintf(namefile, outputjpgDir, 'cross-correlation');
saveas(gcf,outputjpgname);
hold off;
