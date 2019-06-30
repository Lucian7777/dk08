function [data] = readDataBySectionV2_2(SECTION)
load('D:\matlab2016\bin\data_03.mat');
dataIndex = find(data_03(:,1) == SECTION);
dataSection = data_03(dataIndex,:);
dataSection(:,1) = [];
dataSection = sortrows(dataSection,1);
TIMEINMINUTE = 6.9444445e-4;
for i = 2 : size(dataSection,1)
    if (abs(dataSection(i,2) - dataSection(i-1,2)) > 5)
        dataSection(i,2) = dataSection(i-1,2);
    end
end
data = zeros(4320,1);
data(1) = dataSection(1,2);
stdTime = dataSection(1,1);
%index = 2;
for i = 2:4320
    stdTime = stdTime + TIMEINMINUTE;
    stdTimeVe(1:size(dataSection,1),1) = stdTime;
    indexV = abs(dataSection(:,1) - stdTimeVe);
    temp = find(indexV == min(indexV));
    data(i) = dataSection(temp(1),2);
    dataSection(temp(1),1) = 0; %设标志，避免采到同一点
end
% while i < 4320 && index < size(dataSection,1)
%     stdTime = stdTime + TIMEINMINUTE;
%     while (abs(dataSection(index) - stdTime) > 1e-5 && dataSection(index) - stdTime < TIMEINMINUTE)
%         index = index + 1;
%     end
%     data(i) = dataSection(index,2);
%     stdTime = dataSection(index,1);
%     index = index + 1;
%     i = i + 1;
% end
save(num2str(SECTION),'data');