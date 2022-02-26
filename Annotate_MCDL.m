close all;clear all;
path = [pwd,'/Inputs/'];
[filename, pathname] = uigetfile('*.mat', 'Select the result File');
mat_file = fullfile(pathname, filename);
load(mat_file);
%%
use_my_threshold = false;
my_threshold = 0.4;
%% If Features are raw, below flag should be true
need_normalize = false;
%% To Run this script, please put .mat file in the Inputs folder
% Library mex files are compiled for win 64. To run in another operating
% system, you should compile spams-matlab library
% path = [pwd,'/Inputs/'];
%%
warning off;
addpath(genpath('spams-matlab'));
addpath(genpath('measures'));
%%
if(use_my_threshold)
    threshold = my_threshold;
else
    threshold = result.threshold;
end
disp(['Threshold Value for Prediction = ', num2str(threshold)]);
%% Assign train and test from result structure
train = result.train;
test = result.test;
%% Normalization
%% Dimensionally Reduction & Normalization
if(size(train, 2)~=pcaDim)
    pca_map.mean = mean(train);
    train = bsxfun(@minus,train,pca_map.mean);
    test = bsxfun(@minus,test,pca_map.mean);
    [pca_map.P, ~] = pca(train,'NumComponents', pcaDim); %% PCA Transformed Data
    train = train*pca_map.P;
    test = test*pca_map.P;
    clear pcaBasis;
end
train = normc(single(train'));
test = normc(single(test'));
%% Test Similarity and Performance Computing
alphaTest = mexLasso(test,result.D_Input, result.param);
test_similarity = result.D_Label*full(alphaTest);
test_predict = test_similarity>threshold;

[precision,recall,F1]=f1Computing(result.testAnnotation, test_predict);
disp([' Test Precision = ', num2str(precision), '   Recall = ', num2str(recall), '   F1 = ', num2str(F1)]);

%% Train Similarity and Performance Computing
alphaTrain = mexLasso(train,(result.D_Input), result.param);
train_similarity = result.D_Label*full(alphaTrain);
train_predict = train_similarity>threshold;

[precision,recall,F1]=f1Computing(result.trainAnnotation, train_predict);
disp([' Train Precision = ', num2str(precision), '   Recall = ', num2str(recall), '   F1 = ', num2str(F1)]);


