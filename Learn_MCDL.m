%% settings
beta0 = 1.0;
max_w = 5;
w_step = 0.05;
%%
N = size(trainLabel, 2);
T = size(trainLabel, 1);
N_pos = sum(trainLabel>0.5);
%% iteration count
if (isfield(params,'iternum'))
    iternum = params.iternum;
else
    iternum = 15;
end

if (isfield(params,'regL'))
    regL_sqrt = sqrt(params.regL/T);
else
    regL_sqrt = 1/mean(sqrt(sum(trainLabel)));
end

if (isfield(params,'regW'))
    regW = params.regW;
else
    regW = 0.01;
end

if (isfield(params,'dictsize'))
    K = params.dictsize;
else
    disp('Please pass the dictionary size...');
    return;
end
%%
paramLasso.K = K;
paramLasso.lambda = beta0;
paramLasso.mode = 0;
paramLasso.modeD = 0;
paramLasso.modeParam = 0;
paramLasso.L = 20;
paramLasso.iter= 100;  %
paramLasso.pos = true; % for sparse coding
paramLasso.posAlpha = true;
%% coordinate descent parameter
rho = 0.1*sqrt(K);
%%
% Initialization of Visual Prototypes
disp('Initializing visual prototypes using k-means.... ');
reg_coupled = regL_sqrt;
XY = [train; reg_coupled*trainLabel];
[~, XY] = kmeans(XY', paramLasso.K, 'MaxIter', 500);
D_X = XY(:, 1:size(train, 1));
clear XY;
D_X = normc(D_X');
%% Initialize Visual Prototypes
dim = size(train, 1);
eps_alpha = 0.001;
for iter = 1:paramLasso.iter
    disp(['Iteration = ',num2str(iter)]);
    alphaTrain = single(full(mexLasso(train, D_X, paramLasso)));
    x_res = train - D_X*alphaTrain;
    perm = randperm(K);   
    for j = 1:K
        k = perm(mod(j-1, K)+1);
        members_k = find(abs(alphaTrain(k, :))>eps_alpha);
        if isempty(members_k)
            continue;
        end
        alpha_k = alphaTrain(k, members_k);
        x_res(:, members_k) = x_res(:, members_k) + D_X(:, k)*alpha_k;
        coefs_num_k = alpha_k;
        sum_denum_k = sum(coefs_num_k.*alpha_k);
        if(sum_denum_k>eps)
            d_x = (x_res(:, members_k)*coefs_num_k')./sum_denum_k;
            d_x = d_x/norm(d_x);
            D_X(:, k) = d_x;
        else
            d_x = D_X(:, k);
        end
        x_res(:, members_k) = x_res(:, members_k) - d_x*alpha_k;
    end
    clear x_res;
end
%% Initialization of Label Prototypes
alphaTrain = single(full(mexLasso(train, D_X, paramLasso)));
Y_signed = single(trainLabel*2-1);
YM = trainLabel;
YM(trainLabel>0.5) = max(1, Tau+C);
sum_denum = sum(alphaTrain'.^2)+eps;
D_Y = bsxfun(@rdivide, YM*alphaTrain', sum_denum);
D_Y(:,:) = min(max(D_Y, 0), max_w);
clear YM;
clear sum_denum;
%% %%%%%%%%%%%%%%%  Main Loop  %%%%%%%%%%%%%%%%%
eps_alpha = 0.001;
for iter = 1:iternum
    disp(['Iteration = ',num2str(iter)]);
    % Sparse Coding using Marginalized Lasso (Fast Version)
    reg_coupled = regL_sqrt;
    D_MIX = [D_X; reg_coupled*D_Y];
    norm_coupled = max(sqrt(sum(D_MIX.^2)), eps);
    D_MIX = D_MIX./norm_coupled;
    C_W = repmat(1./norm_coupled', [1 N]);
    
    for lars_t=1:4
        Y_hat = D_Y*alphaTrain;
        % Change YY
        Y_Dist = Y_signed.*(Y_hat-Tau);
        true_margin = Y_Dist>=C;
        false_margin = Y_Dist<C;
        clear Y_Dist;
        YM = zeros(size(Y_signed));
        YM(true_margin) = Y_hat(true_margin);
        YM(false_margin) = max(Tau+(Y_signed(false_margin)).*C, 0);
        XY = [train; reg_coupled*YM];
        
        alphaTrain = single(full(mexLassoWeighted(XY,(D_MIX), C_W, paramLasso)));
        alphaTrain = alphaTrain./norm_coupled';
        alphaTrain(alphaTrain<=eps_alpha) = 0;
    end
    
    clear D_MIX;
    clear XY;
    clear C_W;
    clear YM;
    clear XY;
    clear true_margin;
    clear false_margin;
    
    % Optimize Label Dictionary(Coordinate Descent with Warm Restart)
    if validation
        alphaVal = single(full(mexLasso(val, D_X, paramLasso)));
    end
    
    Y_hat = D_Y*alphaTrain;
    x_res = train - D_X*alphaTrain;
    perm = randperm(K);
    %%    
    for j = 1:K
        k = perm(mod(j-1, K)+1);
        members_k = find(abs(alphaTrain(k, :))>eps_alpha);
        if isempty(members_k)
            D_Y(:, k) = 0;
            continue;
        end

        signed_l_k = Y_signed(:, members_k);
        alpha_k = alphaTrain(k, members_k);
        y_hat_k = Y_hat(:, members_k);
        
        dy_k_cur = D_Y(:, k);
        kisi = max(C-signed_l_k.*(y_hat_k-Tau), 0).^2;  
        loss_cur = sum(kisi, 2) + regW*dy_k_cur;  
        
        % optimize loss value
        dy_k_new = dy_k_cur;
        loss_new = loss_cur;
        y_hat_k = y_hat_k - dy_k_cur*alpha_k;
        
        rho_iter = rho*iter;
        range = 0:w_step:max_w;
        for d_j_k = range
            kisi = max(C-signed_l_k.*(y_hat_k+d_j_k*alpha_k-Tau), 0).^2;
            loss = sum(kisi, 2) + regW*d_j_k + rho_iter*(d_j_k-dy_k_cur).^2;
            replace_w = loss<loss_new;
            dy_k_new(replace_w) = d_j_k;
            loss_new(replace_w) = loss(replace_w);
        end

        D_Y(:, k) = dy_k_new;
        Y_hat(:, members_k) = y_hat_k + dy_k_new*alpha_k;
        if(mod(j, floor(K/2))==1)
            kisi = max(C-(2*trainLabel-1).*(D_Y*alphaTrain-Tau), 0).^2;
            loss_new = sum(kisi, 2) + regW*sum(D_Y, 2);
            disp(['* optimizaing of prototype ', num2str(j)]);
            disp(['   Average Loss for Labels = ', num2str(mean(loss_new))]);
            disp(['   Average Loss for Images (No Weight) = ', num2str(mean(sum(x_res.^2)))]);
            if validation
                sim_y = (D_Y*alphaVal);
                pred = sim_y>(Tau);
                [Precision_i, Recall_i, F1_i] = f1Computing(valLabel, pred);
                disp(['   Validation Label Precision = ', num2str(Precision_i), '  Recall = ', num2str(Recall_i), '  F-measure = ', num2str(F1_i)]);
                sim_y = (D_Y*alphaTrain);
                pred = sim_y>(Tau);
                TP = numel(find(pred & trainLabel>0.5));
                FP = numel(find(pred & trainLabel< 0.5));
                TN = numel(find(~pred & trainLabel< 0.5));
                FN = numel(find(~pred & trainLabel> 0.5));
                [Precision_i, Recall_i, F1_i] = f1Computing(trainLabel, pred);
                disp(['   Train Label Precision = ', num2str(Precision_i), '  Recall = ', num2str(Recall_i), '  F-measure = ', num2str(F1_i)]);
                disp(['   W l1-norm =', num2str(sum(sum(D_Y)))]);
                clear sim_y;
                clear pred;
            end
        end
        x_res(:, members_k) = x_res(:, members_k) + D_X(:, k)*alpha_k;
        N_pos_k = N_pos(1, members_k);  
        coefs_num_k = N_pos_k.*alpha_k;
        sum_denum_k = sum(coefs_num_k.*alpha_k);
        if(sum_denum_k>eps)
            d_x = (x_res(:, members_k)*coefs_num_k')./sum_denum_k;
            d_x = d_x/norm(d_x);
            D_X(:, k) = d_x;
        else
            d_x = D_X(:, k);
        end
        x_res(:, members_k) = x_res(:, members_k) - d_x*alpha_k;
    end
    % scaling projection
    for t=1:T
        d_t = D_Y(t, :);
        y_hat_t = d_t*alphaTrain;
        signed_t = Y_signed(t, :);
        kisi = max(C-signed_t.*(y_hat_t-Tau), 0).^2;
        loss_w = regW*sum(d_t);
        loss_new = sum(kisi, 2) + loss_w;
        best_scale = 1;
        range = 0.1:0.01:max_w/max(d_t);
        for scale = range
            kisi = max(C-signed_t.*(scale*y_hat_t-Tau), 0).^2;
            loss = sum(kisi, 2) + scale*loss_w;
            if (loss<loss_new)
                loss_new = loss;
                best_scale = scale;
            end
        end
        D_Y(t, :) = best_scale*d_t;
    end
    clear d_t;
    clear y_hat_t;
    clear kisi;
    clear Y_hat;
    clear x_res;
    clear signed_l_k;
    clear bias;
    
    disp(' optimizaing labels was finished ...');
    % D_Y(abs(D_Y)<1.0e-6) = 0;
    %% Print training Info
    alphaTrainTmp = single(full(mexLasso(train, D_X, paramLasso)));
    info = sprintf('Iteration %d / %d complete', iter, iternum);
    disp(info);
    train_similarity = D_Y*alphaTrainTmp;
    [thOptimals(iter), F1s(iter), precisions(iter), recalls(iter)] = findBestConstantThreshold( train_similarity, trainLabel, (Tau-C)/2, Tau);
    disp([' [Train] Precision = ', num2str(precisions(iter)), '   Recall = ', num2str(recalls(iter)), '   F1 = ', num2str(F1s(iter))]);
    threshold = thOptimals(iter);
    clear train_similarity;
    clear alphaTrainTmp;
    
    %% Performance Computing for Trace
    if validation
        alphaVal = single(full(mexLasso(val, D_X, paramLasso)));
        val_similarity = D_Y*alphaVal;
        val_predict = val_similarity>threshold;
        [precision, recall, F1]=f1Computing(valLabel, val_predict);
        disp([' [Validation] Precision = ', num2str(precision), '   Recall = ', num2str(recall), '   F1 = ', num2str(F1)]);
        clear alphaVal;
        clear val_similarity;
        clear val_predict;
    end
end
clear Y_signed;
disp(' optimizaing dictionaries were finished ...');
%% Print training Info
alphaTrain = single(full(mexLasso(train, D_X, paramLasso)));
info = sprintf('Iteration %d / %d complete', iter, iternum);
disp(info);
train_similarity = D_Y*alphaTrain;
[thOptimals(iter),F1s(iter),precisions(iter),recalls(iter)] = findBestConstantThreshold( train_similarity, trainLabel, (Tau-C)/2, Tau);
disp([' [Train] Final Precision = ', num2str(precisions(iter)), '   Recall = ', num2str(recalls(iter)), '   F1 = ', num2str(F1s(iter))]);
threshold = thOptimals(iter);
clear train_similarity;


