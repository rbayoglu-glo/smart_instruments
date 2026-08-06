%{
    Cadaver Lab - Tracking
    NDI_Localizer Data Formatting
    2/3/22
    David Leff, SPE - dleff@globusmedical.com, x1853

    Reads the NDI_Localizer.xlsx file (imported csv, replacing headers wtih
    one row with headers)
    Separates tools and strays, and exports to NDI_Localizer_Formatted.xlsx
    (sheets labeled with toolIDs, followed by strays)
%}

%% FILE READING AND BASIC DATA FORMATTING

%Reads the NDI_Localizer file
clear; clc; close all
[m,txt,rw]=xlsread('NDI_Localizer.xlsx');

%clearvars -except m txt rw

%Removes SH poses
%{
m(2:3:size(m,1),:)=[];
txt(2:3:size(txt,1),:)=[];
rw(2:3:size(rw,1),:)=[];
%}

%Identifies each tool ID
toolIDs = string([]);
for i = 2:size(txt,1)   %Each data line
    if m(i-1,2)==0  %Strays data line
        
    elseif m(i-1,2)==1||m(i-1,2)==2 %SH pose or NDI pose
        for j = 4:9:(m(i-1,3)-1)*9+4    %Each tool count
            if txt(i,j)~=""&&sum(toolIDs==txt{i,j})==0
                toolIDs(end+1) = txt(i,j);
                %fprintf('%s\n',toolIDs(end)); fprintf('%i\n',i,j)
            end
        end
    end
end

%{
%Archived embodiment
toolIDs = string([]);
for i = 4:3:size(txt,1)
    for j = 4:8:size(txt,2) %Each NDI pose line
        if txt(i,j)~=""&&sum(toolIDs==txt{i,j})==0
            toolIDs(end+1) = txt(i,j);
            %fprintf('%s\n',toolIDs(end)); fprintf('%i\n',i,j)
        end
    end
end
%}

%Separates tool data
tool1 = []; tool2 = []; tool3 = []; tool4 = []; tool5 = []; tool6 = [];
tool7 = []; tool8 = []; tool9 = []; tool10 = []; tool11 = [];
for i = 2:size(m,1)	%Each data line
    if m(i,2)==0  %Strays data line
        
    elseif m(i,2)==1||m(i,2)==2 %SH pose or NDI pose
        for j = 4:9:(m(i,3)-1)*9+4    %Each tool count
            switch txt{i+1,j}
                case ""

                case toolIDs(1)
                    tool1(end+1,:) = [m(i,1) m(i,j+2:j+8)]; %Timestamp and quaternion
                case toolIDs(2)
                    tool2(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(3)
                    tool3(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(4)
                    tool4(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(5)
                    tool5(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(6)
                    tool6(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(7)
                    tool7(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(8)
                    tool8(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(9)
                    tool9(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(10)
                    tool10(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                case toolIDs(11)
                    tool11(end+1,:) = [m(i,1) m(i,j+2:j+8)];
                otherwise
                    fprintf('Error, tool not found\n')
                    fprintf('%i\n',i,j)
            end
        end
    end
end

%{
%Archived embodiment
%Separates tool data
tool1 = []; tool2 = []; tool3 = []; tool4 = []; tool5 = []; tool6 = [];
tool7 = []; tool8 = []; tool9 = []; tool10 = []; tool11 = [];
for i = 4:3:size(txt,1) %Each NDI pose line
    for j = 4:8:size(txt,2) %Each NDI pose line entry
        switch txt{i,j}
            case ""

            case toolIDs(1)
                tool1(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)]; %Timestamp and quaternion
            case toolIDs(2)
                tool2(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(3)
                tool3(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(4)
                tool4(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(5)
                tool5(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(6)
                tool6(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(7)
                tool7(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(8)
                tool8(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(9)
                tool9(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(10)
                tool10(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            case toolIDs(11)
                tool11(end+1,:) = [m(i-1,1) m(i-1,j+1:j+7)];
            otherwise
                fprintf('Error, tool not found\n')
                fprintf('%i\n',i,j)
        end
    end
end
%}

%Separates stray data
strays=[];
for i = 1:3:size(m,1)   %Each strays row
    for j = 6:3:m(i,5)*3+5  %Each stray designated in stray count index
        strays(end+1,:) = [m(i,1) m(i,j:j+2)];  %Timestamp and coordinates
    end
end

%Calculates the maximum number of strays
[~,~,ix]=unique(strays(:,1));
strays_max = max(accumarray(ix,1));

%Writes tool and strays data to file
%NOTE: Manually label tab names with toolIDs, add/remove tool tabs
%NOTE: File paths frequently too long, change directory
xlswrite('NDI_Localizer_Formatted.xlsx',tool1,1)
xlswrite('NDI_Localizer_Formatted.xlsx',tool2,2)
xlswrite('NDI_Localizer_Formatted.xlsx',tool3,3)
xlswrite('NDI_Localizer_Formatted.xlsx',tool4,4)
xlswrite('NDI_Localizer_Formatted.xlsx',tool5,5)
xlswrite('NDI_Localizer_Formatted.xlsx',tool6,6)
xlswrite('NDI_Localizer_Formatted.xlsx',tool7,7)
xlswrite('NDI_Localizer_Formatted.xlsx',tool8,8)
xlswrite('NDI_Localizer_Formatted.xlsx',tool9,9)
xlswrite('NDI_Localizer_Formatted.xlsx',tool10,10)
xlswrite('NDI_Localizer_Formatted.xlsx',tool11,11)
xlswrite('NDI_Localizer_Formatted.xlsx',strays,12)

%% ALTERNATIVE FILE READING AND BASIC DATA FORMATTING

%m=readtable('NDI_Localizer.csv');

%{
ds = tabularTextDatastore('NDI_Localizer.csv');
ds.ReadSize = 3;
while hasdata(ds)
    dsframe = read(ds);
end
%}

%{
%Separate Strays Data
for i = 1:2:size(m,2)
    strays_tmp(end,:)=m(i,:);
    for j = 1:size(strays,2)
        if ischar(strays_tmp(j))
    end
end
%}

