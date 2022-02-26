function [ th, maxF1,maxprecision,maxrecall ] = findBestConstantThreshold( tainLabelSimilarity, trainAnnotation, LowerBound, UpperBound)
thS = [LowerBound:0.005:UpperBound];
outLabel = tainLabelSimilarity;
realLabel = trainAnnotation>0.5;
for index = 1:numel(thS);
    guess = outLabel>thS(index);
    [precision,recall,F1]=f1Computing(realLabel,guess);
    precisions(index)=precision;
    recalls(index)=recall;
    F1s(index) = F1;
end
[maxF1,index]=max(F1s);
maxprecision=precisions(index);
maxrecall=recalls(index);
th = thS(index);
end
