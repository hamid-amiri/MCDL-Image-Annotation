function [precision,recall,F1]=f1Computing(testAnnotation,guess)

TP = sum(guess==1 & testAnnotation==1, 2);
FP = sum(guess==1 & testAnnotation==0, 2);
FN = sum(guess==0 & testAnnotation==1, 2);

P = TP ./ max(TP+FP, eps); %%! +eps for NaN prevention 
R = TP ./ max(TP+FN, eps);

precision = mean(P);
recall = mean(R);
F1 = (2.*precision.*recall) ./ max(precision+recall, eps);

end