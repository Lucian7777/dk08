function [data]=readDataBySection(Section)
load('D:\matlab2016\bin\data_03.mat');
dataIndex=find(data_03(:,1)==Section);
dataSection=data_03(dataIndex,:);
dataSection=sortrows(dataSection,2);
% deletImproperIndex=find(dataSection(:,3)>dataFirst+40);
dataDif=zeros(size(dataSection,1),1);
dataDif(1)=1;
for i=2:size(dataSection,1)
    dataDif(i)=dataSection(i,2)-dataSection(i-1,2);
end
dataRepetition=find(dataDif<3e-4);
dataSection(dataRepetition,:)=[];
% dataDif=zeros(size(dataSection,1),1);
% k=size(dataSection,1)-1;
% for i=2:k
%     while abs(dataSection(i,3)-dataSection(i-1,3))>10
%         dataSection(i,:)=[];
%         k=k-1;
%     end
% end
% dataS=find(dataDif>10);
% data=dataSection;
data=zeros(4320,1);
i=2;
n=1;
while n<4321&&i<size(dataSection,1)
if abs(dataSection(i,3)-dataSection(i-1,3))<10
    data(n)=dataSection(i-1,3);
    n=n+1;
    i=i+1;
else
    i=i+1;
end
end
data(find(data==0))=[];
dataFirst=data(1);
deletImproperIndex=find(data<dataFirst-40);
data(deletImproperIndex,:)=[];
plot(data);
save(num2str(Section),'data');