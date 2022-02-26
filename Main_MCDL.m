close all;
clear all;
clc;
warning off;
rng('default');
%% please refer to http://thoth.inrialpes.fr/people/mairal/spams/ for sapams-matlab
addpath(genpath('spams-matlab'));
addpath(genpath('measures'));
%%
save_all = true;
%%
path = ['Data/'];
dataset = 'Iaprtc12'; %'Iaprtc12','Espgame','Flickr60k','Flickr125k'
featureName = 'deepFeature_dense161';
pca_dim = 200;
dictSizes = [4000];
itterations = 15; % Dictionary Learning Iterations

%% Tunning Parameters
C   = 0.5;
Tau = 0.25 + C;
reg_Ls = [0.5].^2;
reg_Ws = [0.15, 0.2];
%% read dataset
load (strcat(path,'testAnnotation',dataset,'.mat'));
load (strcat(path,'trainAnnotation',dataset,'.mat'));
tempFeature = load(strcat(path, featureName, dataset, '.mat'));
eval(sprintf('%s=double(%s);','tempFeature',['tempFeature.',featureName]));
train = tempFeature(1:size(trainAnnotation,1),:,:);
test = tempFeature(size(trainAnnotation,1)+1:end,:,:);
clear tempFeature;
numLabels = size(trainAnnotation,2);
%% Dimensionally Reduction & Normalization
if(size(train, 2)~=pca_dim)
    % PCA Dimensionally Reduction
    [~, pca_map.mean, pca_map.sigma] = zscore(train);
    train = bsxfun(@minus,train,pca_map.mean);
    test = bsxfun(@minus,test,pca_map.mean);
    train = bsxfun(@rdivide,train,pca_map.sigma);
    test = bsxfun(@rdivide,test,pca_map.sigma);
    [pca_map.P, ~, ~] = pca(train, 'NumComponents', pca_dim); %% PCA Transformed Data
    train = train*pca_map.P;
    test = test*pca_map.P;
    clear pcaBasis;
end
% Normalize after PCA
train = normc(single(train'));
test = normc(single(test'));
trainLabel = single(trainAnnotation');
clear trainAnnotation;
testLabel = single(testAnnotation');
clear testAnnotation;
%% learning demo
regW_idx = 1;
resume = false;

for dsize_idx=1:length(dictSizes)
    perf_best = 0;
    params.dictsize =  dictSizes(dsize_idx);
    for regL_idx=1:length(reg_Ls)
        for regW_idx=1:length(reg_Ws)
            mkdir('Results')
            params.regL = reg_Ls(regL_idx);
            params.regW = reg_Ws(regW_idx);
            res_filename = strcat('Results/', dataset, '_', featureName,'_K',num2str(params.dictsize),...
                '_C_', num2str(C), '_L_', num2str(params.regL), '_W_', num2str(params.regW), '_result.mat');
            if resume && exist(res_filename)
                disp('A result file has been found and parameters will be skipped ...')
                continue;
            end
            Learn_MCDL; % save memory to call as an script
            %% Train Similarity
            alphaTrain = single(full(mexLasso(train,(D_X),paramLasso)));
            train_similarity = D_Y*full(alphaTrain);
            clear alphaTrain;
            %% Test Similarity
            tic
            alphaTest = single(full(mexLasso(test,D_X,paramLasso)));
            test_similarity = D_Y*alphaTest;
            toc
            clear alphaTest;
            %% Constant Threshold Learning
            tic
            [ threshold, ~ ] = findBestConstantThreshold( train_similarity, trainLabel, (Tau-C)/2, Tau);
            %% Performance Computing
            train_predict = train_similarity>threshold;
            test_predict = test_similarity>threshold;
            [precision,recall,F1]=f1Computing(testLabel, test_predict);
            disp([' [Test] Final Precision = ', num2str(precision), '   Recall = ', num2str(recall), '   F1 = ', num2str(F1)]);
            toc
            disp(' ===================================================================================');
            %% Save Status
            if F1 > perf_best
                perf_best = F1;
            end
            result.F1 = F1;
            result.precision = precision;
            result.recall = recall;
            result.optimization_regL = reg_Ls(regL_idx);
            result.optimization_regW = reg_Ws(regW_idx);
            if save_all
                result.param = paramLasso;
                result.test = test;
                clear test;
                result.train = train;
                clear train;
                result.train_similarity = train_similarity;
                clear train_similarity;
                result.test_similarity = test_similarity;
                clear test_similarity;
                result.train_predict = train_predict;
                clear train_predict;
                result.test_predict = test_predict;
                clear test_predict;
                result.testAnnotation = testLabel;
                clear testLabel;
                result.trainAnnotation = trainLabel;
                clear trainLabel;
                result.D_Input = D_X;
                clear D_X;
                result.D_Label = D_Y;
                clear D_Y;
                result.threshold = threshold;
            end
            delete res_filename
            save(res_filename,'result');
            clear result;
        end
    end
end