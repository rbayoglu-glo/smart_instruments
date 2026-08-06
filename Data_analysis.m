%{
    Cadaver Lab - Tracking
    Data Analyssi
    2/8/22
    David Leff, SPE - dleff@globusmedical.com, x1853

    Analyzes the NDI_Localizer.xlsx file (output of Data_formatting)
%}

%% FILE READING

%Reads relevant arrays
drb = xlsread('NDI_Localizer_Formatted.xlsx',2);    %DRB in iliac crest
drill = xlsread('NDI_Localizer_Formatted.xlsx',4);  %Reducer array 1
tap = xlsread('NDI_Localizer_Formatted.xlsx',5);    %Extender array
verp = xlsread('NDI_Localizer_Formatted.xlsx',6);   %Verification probe
dilat = xlsread('NDI_Localizer_Formatted.xlsx',8);  %SP Clamp
prob = xlsread('NDI_Localizer_Formatted.xlsx',7);   %Reducer array 2

driv = xlsread('NDI_Localizer_Formatted.xlsx',3);   %Driver array?
purp = xlsread('NDI_Localizer_Formatted.xlsx',9);   %Purple array?
grn = xlsread('NDI_Localizer_Formatted.xlsx',10);   %Green array?

%% DATA FORMATTING

%Transforms quaternions to rotation matrices, stored in row vectors
drb_r = [];
for i = 1:size(drb,1)
    q = drb(i,5:8); %Quaternion components
    drb_r(end+1,:) = [drb(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end
drill_r = [];
for i = 1:size(drill,1)
    q = drill(i,5:8);
    drill_r(end+1,:) = [drill(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end
tap_r = [];
for i = 1:size(tap,1)
    q = tap(i,5:8);
    tap_r(end+1,:) = [tap(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end
dilat_r = [];
for i = 1:size(dilat,1)
    q = dilat(i,5:8);
    dilat_r(end+1,:) = [dilat(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end
prob_r = [];
for i = 1:size(prob,1)
    q = prob(i,5:8);
    prob_r(end+1,:) = [prob(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end
purp_r = [];
for i = 1:size(purp,1)
    q = purp(i,5:8);
    purp_r(end+1,:) = [purp(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end
grn_r = [];
for i = 1:size(grn,1)
    q = grn(i,5:8);
    grn_r(end+1,:) = [grn(i,1:4)...
        2*(q(1)^2+q(2)^2)-1 2*(q(2)*q(3)-q(1)*q(4)) 2*(q(2)*q(4)+q(1)*q(3))...
        2*(q(2)*q(3)+q(1)*q(4)) 2*(q(1)^2+q(3)^2)-1 2*(q(3)*q(4)-q(1)*q(2))...
        2*(q(2)*q(4)-q(1)*q(3)) 2*(q(3)*q(4)+q(1)*q(2)) 2*(q(1)^2+q(4)^2)-1];
end

%% DATA ANALYSIS

%Designates timestamp ranges of interest for analysis
%NOTE: Not used, data noisy - see i_drb_rs for index range analyzed below
tstmps1 = [1.96545 1.96560; 1.96560 1.96565; 1.96565 1.96590].*10^11;    %First sequence
tstmps2 = [1.96765 1.967745].*10^11;	%Seccond sequence

%Conducts analysis over the designated DRB index
i_grn_r = []; i_prob_r = []; i_dilat_r = [];
h_drb_dilat = []; h_drb_red1 = []; h_drb_red2 = [];
hd_drb_dilat = []; hd_drb_red1 = []; hd_drb_red2 = [];
td_drb_dilat = []; td_drb_red1 = []; td_drb_red2 = [];
i_drb_rs = [101318 103916];  %DRB starting and ending index for seccond sequence
td_tstmp = [];
for i_drb_r = i_drb_rs(1):i_drb_rs(2)
    %Find indices at the timestamp
    i_grn_r = find(grn_r(:,1)==drb_r(i_drb_r));
    i_prob_r = find(prob_r(:,1)==drb_r(i_drb_r));
    i_dilat_r = find(dilat_r(:,1)==drb_r(i_drb_r));
    
    if isempty(i_grn_r)||isempty(i_prob_r)||isempty(i_dilat_r)
        %No simulatneous data for all arrays - drop frame
    else
        %Assign timestamp
        td_tstmp(end+1,:) = drb_r(i_drb_r,1);
        
        %Assign rotation matrices
        h_c_drb = [drb_r(i_drb_r,5:7) drb_r(i_drb_r,2);...
            drb_r(i_drb_r,8:10) drb_r(i_drb_r,3);...
            drb_r(i_drb_r,11:13) drb_r(i_drb_r,4);
            0 0 0 1];
        h_c_grn = [grn_r(i_grn_r,5:7) grn_r(i_grn_r,2);...
            grn_r(i_grn_r,8:10) grn_r(i_grn_r,3);...
            grn_r(i_grn_r,11:13) grn_r(i_grn_r,4);
            0 0 0 1];
        h_c_prob = [prob_r(i_prob_r,5:7) prob_r(i_prob_r,2);...
            prob_r(i_prob_r,8:10) prob_r(i_prob_r,3);...
            prob_r(i_prob_r,11:13) prob_r(i_prob_r,4);
            0 0 0 1];
        h_c_dilat = [dilat_r(i_dilat_r,5:7) dilat_r(i_dilat_r,2);...
            dilat_r(i_dilat_r,8:10) dilat_r(i_dilat_r,3);...
            dilat_r(i_dilat_r,11:13) dilat_r(i_dilat_r,4);
            0 0 0 1];
        
        %Calculate and store transformation matrices WRT DRB
        h_drb_dilat(end+1,:,:) = h_c_drb\h_c_dilat;
        h_drb_red1(end+1,:,:) = h_c_drb\h_c_grn;
        h_drb_red2(end+1,:,:) = h_c_drb\h_c_prob;
        
        %Calculate transformation from initial index
        hd_drb_dilat(end+1,:,:) = squeeze(h_drb_dilat(end,:,:))\...
            squeeze(h_drb_dilat(1,:,:));
        hd_drb_red1(end+1,:,:) = squeeze(h_drb_red1(end,:,:))\...
            squeeze(h_drb_red1(1,:,:));
        hd_drb_red2(end+1,:,:) = squeeze(h_drb_red2(end,:,:))\...
            squeeze(h_drb_red2(1,:,:));
        
        %Calculate euler angels from initial index
        %eul_dilat(end+1,:) = rotm2eul(hd_drb_dilat(end,1:3,1:3));
        %NOTE: Get robotics system toolbox for rotm2eul
        %Euler angles manually calculated below
        rd_drb_dilat = squeeze(hd_drb_dilat(end,1:3,1:3));
        tx = atan2(rd_drb_dilat(3,2),rd_drb_dilat(3,3));
        ty = atan2(-rd_drb_dilat(3,1),sqrt(rd_drb_dilat(3,2)^2+rd_drb_dilat(3,3)^2));
        tz = atan2(rd_drb_dilat(2,1),rd_drb_dilat(1,1));
        td_drb_dilat(end+1,:) = [tx ty tz];
        rd_drb_red1 = squeeze(hd_drb_red1(end,1:3,1:3));
        tx = atan2(rd_drb_red1(3,2),rd_drb_red1(3,3));
        ty = atan2(-rd_drb_red1(3,1),sqrt(rd_drb_red1(3,2)^2+rd_drb_red1(3,3)^2));
        tz = atan2(rd_drb_red1(2,1),rd_drb_red1(1,1));
        td_drb_red1(end+1,:) = [tx ty tz];
        rd_drb_red2 = squeeze(hd_drb_red2(end,1:3,1:3));
        tx = atan2(rd_drb_red2(3,2),rd_drb_red2(3,3));
        ty = atan2(-rd_drb_red2(3,1),sqrt(rd_drb_red2(3,2)^2+rd_drb_red2(3,3)^2));
        tz = atan2(rd_drb_red2(2,1),rd_drb_red2(1,1));
        td_drb_red2(end+1,:) = [tx ty tz];
    end
end
td_tstmp=td_tstmp-td_tstmp(1);

%Calculate DRB-space cordinates (just x-y-z)
dilat_drb = [];
for i = 1:size(dilat,1)
    i_drb = find(drb(:,1)==dilat(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        dilat_drb(end+1,:) = [dilat(i,1) dilat(i,2:4)-drb(i_drb,2:4)];
    end
end
tap_drb = [];
for i = 1:size(tap,1)
    i_drb = find(drb(:,1)==tap(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        tap_drb(end+1,:) = [tap(i,1) tap(i,2:4)-drb(i_drb,2:4)];
    end
end
drill_drb = [];
for i = 1:size(drill,1)
    i_drb = find(drb(:,1)==drill(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        drill_drb(end+1,:) = [drill(i,1) drill(i,2:4)-drb(i_drb,2:4)];
    end
end
driv_drb = [];
for i = 1:size(driv,1)
    i_drb = find(drb(:,1)==driv(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        driv_drb(end+1,:) = [driv(i,1) driv(i,2:4)-drb(i_drb,2:4)];
    end
end
purp_drb = [];
for i = 1:size(purp,1)
    i_drb = find(drb(:,1)==purp(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        purp_drb(end+1,:) = [purp(i,1) purp(i,2:4)-drb(i_drb,2:4)];
    end
end
grn_drb = [];
for i = 1:size(grn,1)
    i_drb = find(drb(:,1)==grn(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        grn_drb(end+1,:) = [grn(i,1) grn(i,2:4)-drb(i_drb,2:4)];
    end
end
verp_drb = [];
for i = 1:size(verp,1)
    i_drb = find(drb(:,1)==verp(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        verp_drb(end+1,:) = [verp(i,1) verp(i,2:4)-drb(i_drb,2:4)];
    end
end
prob_drb = [];
for i = 1:size(prob,1)
    i_drb = find(drb(:,1)==prob(i,1)); %Find DRB index
    if isempty(i_drb)   %No corresponding DRB data, exclude
        
    else    %Transfrom x-y-z coordinates to DRB space
        prob_drb(end+1,:) = [prob(i,1) prob(i,2:4)-drb(i_drb,2:4)];
    end
end

%% PRELIMINARY PLOTTING

%{
%Plot overall Z
figure
plot(drb(:,1),drb(:,4),'k.')
hold on
plot(dilat(:,1),dilat(:,4),'r.')
plot(tap(:,1),tap(:,4),'b.')
plot(drill(:,1),drill(:,4),'ko')
plot(driv(:,1),driv(:,4),'c.')
plot(purp(:,1),purp(:,4),'m.')
plot(grn(:,1),grn(:,4),'g.')
plot(verp(:,1),verp(:,4),'bo')
legend('DRB','L3 SP Clamp','Extender','Reducer 1','Driver?','Purple?','Green?','Ver. Pro.')
title('All Tools Z Data')

%Plot screw extender Z
figure
plot(drb(:,1),drb(:,4),'k.')
hold on
plot(dilat(:,1),dilat(:,4),'r.')
plot(tap(:,1),tap(:,4),'b.')
plot(verp(:,1),verp(:,4),'bo')
legend('DRB','L3 SP Clamp','Extender','Ver. Pro.')
title('Screw Extender Tracking')
%}

%Plot drb-z-subtracted overall Z
figure
plot(dilat_drb(:,1),dilat_drb(:,4),'r.')
hold on
plot(tap_drb(:,1),tap_drb(:,4),'b.')
plot(purp_drb(:,1),purp_drb(:,4),'m.')
plot(drill_drb(:,1),drill_drb(:,4),'ko')
plot(grn_drb(:,1),grn_drb(:,4),'g.')
plot(prob_drb(:,1),prob_drb(:,4),'mo')
plot(verp_drb(:,1),verp_drb(:,4),'bo')
plot(driv_drb(:,1),driv_drb(:,4),'c.')
legend('L3 SP Clamp','Extender','Purple/Tap-Extender','Drill-Reducer 1',...
    'Green/Drill-Reducer 1','Probe-Reducer 2','Ver. Pro.','Driver-Unused')
title('All Tools Z Data, DRB Space')
xlabel('Timestamp')
ylabel('Z Coordinate (mm)')

%Plot angular displacements
figure
subplot(2,2,1)
%plot(1:size(td_drb_dilat,1),td_drb_dilat(:,1)*180/pi,'k.')
plot(td_tstmp,td_drb_dilat(:,1)*180/pi,'k.')
hold on
plot(td_tstmp,td_drb_red1(:,1)*180/pi,'r.')
plot(td_tstmp,td_drb_red2(:,1)*180/pi,'b.')
axis([0 9*10^6 -1 5])
legend('L3 SP Clamp','Reducer 1','Reducer 2','Location','northwest')
title('Lordotic Pitch')
xlabel('Local Timestamp')
ylabel('Euler Angle (deg)')
subplot(2,2,2)
plot(td_tstmp,td_drb_dilat(:,2)*180/pi,'k.')
hold on
plot(td_tstmp,td_drb_red1(:,2)*180/pi,'r.')
plot(td_tstmp,td_drb_red2(:,2)*180/pi,'b.')
axis([0 9*10^6 -2 3])
legend('L3 SP Clamp','Reducer 1','Reducer 2','Location','northwest')
title('Coronal Yaw')
xlabel('Local Timestamp')
ylabel('Euler Angle (deg)')
subplot(2,2,3)
plot(td_tstmp,td_drb_dilat(:,3)*180/pi,'k.')
hold on
plot(td_tstmp,td_drb_red1(:,3)*180/pi,'r.')
plot(td_tstmp,td_drb_red2(:,3)*180/pi,'b.')
axis([0 9*10^6 -3 3])
legend('L3 SP Clamp','Reducer 1','Reducer 2','Location','northwest')
title('Axial Roll')
xlabel('Local Timestamp')
ylabel('Euler Angle (deg)')