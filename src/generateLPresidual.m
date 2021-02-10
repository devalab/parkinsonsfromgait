clear all;
close all;
%% Load Dataset
files = dir("../dataset/raw/");
order = 11;
perf = [];
decimationRate = 2;

normalGait = {};
padding = zeros(order,18);
normalGait = cat(1,normalGait,padding);

for i = {files.name}
    i = string(i);
    if strlength(i) == 13 
        if i{1}(3:4) == "Co"
            mat = importdata("../dataset/raw/"+i);
%             mat = mat./std(mat);
            downsampled = {};
            for j = mat
                j = filter(ones(1,decimationRate)/decimationRate,1,j);
%                 j = j/std(j);               
                downsampled =  cat(1,downsampled,j(1:decimationRate:end,1));            
            end
            mat = horzcat(downsampled{:});
            normalGait = cat(1,normalGait,mat(:,2:end));
            normalGait = cat(1,normalGait,padding);
%             break
        end
    end
end

normalGait = vertcat(normalGait{:});
%% Normalize Gait
stds = zeros(18,1);

% for i = 1:18
%     disp(std(normalGait(:,i)));
%     disp(i);
% end 

for i = 1:18
    stds(i,1) = std(normalGait(:,i));
    normalGait(:,i) = normalGait(:,i)./stds(i,1);
end 
% for i = 1:18
%     disp(std(normalGait(:,i)));
% end 
%% Generate LP Coefficients

lpFilters = zeros(18,order+1);

for i = 1:18
    [fil,g] = lpc(normalGait(:,i),order);
    sensorNumber = i;
    lpFilters(sensorNumber,:) = fil;
end

save("../models/lp"+"Order"+num2str(order)+"Decimation"+num2str(decimationRate)+".txt","lpFilters",'-ascii', '-double', '-tabs');

%% Generate LP Residual

files = dir("../dataset/raw/");
% lpFilters = importdata("models/lp.txt");

mkdir("../dataset/processed"+num2str(100/decimationRate)+"/lp_residual/");
mkdir("../dataset/processed"+num2str(100/decimationRate)+"/force/");
% mkdir("./dataset/processed"+num2str(100/decimationRate)+"/features/");

for i = {files.name}
    i = string(i);
    if strlength(i) == 13 
        mat = importdata("../dataset/raw/"+i);
        tic;
        downsampled = {};
        for sensor = 2:19
            j = mat(:,sensor);
            j = filter(ones(1,decimationRate)/decimationRate,1,j);
            j = j./stds(sensor-1,1);               
            downsampled =  cat(1,downsampled,j(1:decimationRate:end,1));            
        end
        mat = horzcat(downsampled{:});
        
        mat_processed = mat;
        for j = 1:18
            mat_processed(:,j) = filter([0 -lpFilters(j,2:end)],1,mat(:,j));
        end      
        
        mat_processed = mat - mat_processed;
        
        toc;
        
        csvwrite("../dataset/processed"+num2str(100/decimationRate)+"/lp_residual/"+i,mat_processed);
        csvwrite("../dataset/processed"+num2str(100/decimationRate)+"/force/"+i,mat);
    end
end

%% Time Processing

time = ones(1,1000);
for i = 1:1000
    mat = randn(12000,18);
    tic;
    downsampled = {};
    for sensor = 1:18
        j = mat(:,sensor);
        j = filter(ones(1,decimationRate)/decimationRate,1,j);
        j = j./stds(sensor,1);               
        downsampled =  cat(1,downsampled,j(1:decimationRate:end,1));            
    end
    mat = horzcat(downsampled{:});

    mat_processed = mat;
    for j = 1:18
        mat_processed(:,j) = filter([0 -lpFilters(j,2:end)],1,mat(:,j));
    end      

    mat_processed = mat - mat_processed;    
%     toc;
    time(1,i) = toc;
end
mean(time)