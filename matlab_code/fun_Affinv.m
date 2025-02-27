function [Detected_powers, Distance, p, normal, noisepower] = fun_Affinv(Y,A,DAS_init,DOAscan,DOA, noise_power)

t_samples = size(Y, 2);
R_hat = Y*Y' / t_samples;
sigma2_n = noise_power;
% p = GetFullCovAffinvEstimation(sigma2_n,DOAscan*pi/180,R_hat);
[p, ~] = fun_Affinv_aux(sigma2_n,A,R_hat,DAS_init);





[pks index]=findpeaks(p, 'sortstr', 'descend');
Numsources =length(DOA);
if length(index) < Numsources
%     warning('Not all peaks detected');
    normal = 0;
    Distance = NaN;
%     p = NaN;
    Detected_powers = NaN;
    noisepower = noise_power;
    return;
end

% ------------ Check whether the detection is right -----
Detected_DOAs = DOAscan(index(1:Numsources));

[Detected_DOAs, IXsort] = sort(Detected_DOAs, 'ascend');
Distance = Detected_DOAs - DOA;
normal = 1; % detection okay
% the powers from large value to small value
Detected_powers = p(index(1:Numsources));
% sort the power according to the DOA
Detected_powers =  Detected_powers(IXsort);



noisepower = sigma2_n;


end
%%
function [p, logger] = fun_Affinv_aux(sigma2_n,A,GammaTensor,DAS_init)
%%
t0 = tic;
D = size(A,2);
[M,~,L] = size(GammaTensor);
%%
% NUM_MAX_ITERS = 1000;
NUM_MAX_ITERS = 1e4;
% eta = 1e-2;
% eta = 1e-5;
% eta = 1e-3;1e-4;1e-3; %%%% 20/10/2024
eta = 1e-3; %% 18/02/2025
eps_pow = 1e-5;
nu = 0;D^2 * 1e-8;
beta = 0.9;
EPS_NORM_CHANGE = 1e-4;

%%
p_init = DAS_init;
% p_init = 0.5*ones(D,1);

% logm_Gamma_LE_mean = 0;
% for l = 1:L
%     currGamma = GammaTensor(:,:,l);
%     logm_currGamma = logm(currGamma);
%     logm_Gamma_LE_mean = logm_Gamma_LE_mean +  logm_currGamma/ L;
% end
% logm_Gamma_LE_mean = (logm_Gamma_LE_mean + logm_Gamma_LE_mean')/2;
% log_p_init = real(dot(A,logm_Gamma_LE_mean * A)');
% p_init = exp(log_p_init) - sigma2_n;
% p_init(p_init < eps_pow) = 0;
% 
% p_init = 0.1 * ones(size(p_init));
% p_init = GetEstimationBySPICE(sigma2_n, vec_theta_grid, GammaTensor);
% p_init(p_init < eps_pow) = 0;

% p_init = GetEstimationByAML(sigma2_n, vec_theta_grid, GammaTensor);

%%
logger = struct;
logger.p_init = p_init;
logger.all_grads = [];
logger.all_ps = [];
logger.loss_func = @LossFunc;
logger.loss_vec = [];
%%
p = p_init;
sum_grad_squared = 0;
iter_best_p = nan; loss_best_p = inf;
iter_prev_best_p = nan; loss_prev_best_p = inf;
averageGrad = []; averageSqGrad = [];
for iter = 1:(NUM_MAX_ITERS+1)
    R = A*diag(p)*A' + sigma2_n*eye(M);
    grad = 0;
    prev_loss = 0;
    for l=1:L
        invsqrtm_R_l = pinv(sqrtm(GammaTensor(:,:,l)));
        Q = invsqrtm_R_l * R * invsqrtm_R_l;
        Q = (Q+Q')/2; % for numerical stability
        [U,Lam,~] = svd(Q);
        lambdas = diag(Lam);

        prev_loss = prev_loss + sum(log(lambdas).^2)/L;
        
        V = abs(U' * invsqrtm_R_l * A).^2;
        b = 2*log(lambdas)./lambdas;
        grad = grad + (1/L)* V'*b;        
    end
    logger.loss_vec(iter) = prev_loss; % remember: loss calculted at iteration i is the loss for p_(i-1)
    
    if (prev_loss < loss_best_p)
        iter_prev_best_p = iter_best_p;
        loss_prev_best_p = loss_best_p;

        iter_best_p = iter-1;
        loss_best_p = prev_loss;
    end
    if (iter > NUM_MAX_ITERS), break; end
    if iter_prev_best_p > 2 && iter_best_p > 2
        measured_change_norm = norm(logger.all_ps(:,iter_best_p) - logger.all_ps(:,iter_prev_best_p))/norm(logger.all_ps(:,iter_prev_best_p));
        if measured_change_norm < EPS_NORM_CHANGE
            break;
        end
    end
    % ----
    grad_regularization = nu*(p >= eps_pow);
    grad = grad + grad_regularization;
    %%
    % sum_grad_squared = beta*sum_grad_squared + (1-beta)*grad.^2;
    % step_size_vec = eta ./ sqrt(1e-7 + sum_grad_squared);    
    % % ----
    % p = p - step_size_vec .* grad;
    %% ----
    learnRate = 1e-3;
    gradDecay = 0.95; 0.75;
    sqGradDecay = 0.95;
    [p,averageGrad,averageSqGrad] = adamupdate(p,grad,averageGrad,averageSqGrad,iter,learnRate,gradDecay,sqGradDecay);
    %%
    p(p < eps_pow) = 0;
    
    %%
    logger.all_grads(:,iter) = grad;
    logger.all_ps(:,iter) = p;
end
%% choose last iter (by loss values)
% WINDOW_LEN_CHOOSE_LAST_ITER = 10;
% num_iters = iter-1;
% check_iters = num_iters-WINDOW_LEN_CHOOSE_LAST_ITER+1:num_iters;
% [~,i_offset] = min(logger.loss_vec(check_iters+1)); % plus 1 for the offset of the loss vec
% chosen_last_iter = check_iters(i_offset);
chosen_last_iter = iter_best_p;

logger.all_grads = logger.all_grads(:,1:chosen_last_iter);
logger.all_ps = logger.all_ps(:,1:chosen_last_iter);
logger.loss_vec = logger.loss_vec(1: (chosen_last_iter+1));
if chosen_last_iter == 0
    p = p_init;
else
    p = logger.all_ps(:,chosen_last_iter);
end

if (false)
    %%
    titleStr = "Affinv loss function";
    pMat = [logger.p_init, logger.all_ps];
    iters_to_display = 0:10:(size(pMat,2)-1);
    loss_vec = logger.loss_vec(iters_to_display+1);%logger.loss_func(sigma2_n,vec_theta_grid,GammaTensor,pMat(:,iters_to_display+1));
    curr_fig = gcf;
    tmp_fig = figure(1111); grid on; hold on; ax = gca;
    plot(ax, iters_to_display,loss_vec,'-o');
    xlabel('iters'); ylabel('loss'); title({titleStr,'loss'});
    % if isfield(logger,'loss_vec')
    %     plot(ax, 0:(length(logger.loss_vec)-1),logger.loss_vec,'--o', 'ButtonDownFcn',@plotClickCallback);
    % end
    figure(curr_fig);
end
disp("AFFINV time = " + toc(t0) + "[sec]");
end
