%% Precision-Recall Compute
addpath('measures');
num_thresholds = size(result.test_similarity, 1);
max_th = 1;
th_step = max_th/(num_thresholds);
prfs = zeros(num_thresholds, 3);
for i = 1:num_thresholds
    th = max_th - th_step*i;
    test_predict = result.test_similarity>=th;
    [precision,recall,F1]=f1Computing(result.testAnnotation, test_predict);
    prfs(i, :) = [precision,recall,F1];
    disp([' Train Precision = ', num2str(precision), '   Recall = ', num2str(recall), '   F1 = ', num2str(F1)]);
end