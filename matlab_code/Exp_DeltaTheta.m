function Exp_DeltaTheta(N, cohr_flag, Large_Scale_Flag)
rng(42);
Flag_save_fig = true;
M = 12;
timestamp = datestr(now, 'dd-mm-yyyy_HH-MM-SS');
str_indp_cohr = 'indp';
if (cohr_flag), str_indp_cohr = 'cohr'; end
results_dir = ['Exp_DeltaTheta_' timestamp '_' str_indp_cohr '_M' num2str(M) '_N' num2str(N) '_L' num2str(Large_Scale_Flag) '/' ];



if ~exist(results_dir, 'dir') && Flag_save_fig
    mkdir(results_dir);
end


if Large_Scale_Flag
    disp('==== Large SCALE MC tests Running...');
    NUM_MC =  500;
    VEC_DELTA_THETA= 2:10;
else
    disp('=========== SMALL SCALE MC tests@@@ !!! =======');
    NUM_MC =  50;
    VEC_DELTA_THETA= 2:10;
end
SNR = 0;
algo_list = ["PER", "SPICE", "SAMV", "AFFINV"];

num_algos = length(algo_list);

SE_mean = zeros(length(VEC_DELTA_THETA), num_algos);
Failing_rate = zeros(length(VEC_DELTA_THETA), num_algos);


CRBlist = zeros(1, length(VEC_DELTA_THETA));

%%
PowerDOAdB = [3;4];%[5; 3]; % in dB
firstDOA = 35.11;
%%
for delta_theta_ind = 1:length(VEC_DELTA_THETA)
    delta_theta = VEC_DELTA_THETA(delta_theta_ind);
    disp(['=== Computing DeltaTheta == ' num2str(delta_theta)]);
    DOA = [firstDOA, firstDOA + delta_theta];

    [SE_mean_perAlgo, Failing_rate_perAlgo, CRB_val] = ...
        ComputeAlgosStdErr(algo_list, NUM_MC, SNR, N, M, cohr_flag, PowerDOAdB, DOA);
    
    SE_mean(delta_theta_ind, :) = SE_mean_perAlgo;
    Failing_rate(delta_theta_ind,:) = Failing_rate_perAlgo;
    CRBlist(delta_theta_ind) = CRB_val;
    
end


if Flag_save_fig
    save([results_dir '/Algos_Data.mat' ]);%, 'SNRSetdB', 'SE_mean', 'Failing_rate', 'algo_list' );
end

%% ==== Plot figures =========
colorSet={'r-->', 'm--p','b-^','g--s', 'k--'};


% plot SE figure;
h2 = figure;
for i_algo = 1: num_algos
    prepare = sqrt(SE_mean(:,i_algo).' + eps);
    semilogy(VEC_DELTA_THETA, prepare, colorSet{i_algo},'DisplayName',algo_list(i_algo));
    hold on;
end

% -----CRB ----
semilogy(VEC_DELTA_THETA, sqrt(CRBlist), colorSet{num_algos+1},'DisplayName','CRB'); 

xlabel('$\Delta \theta$ (degrees)','Interpreter','latex');
ylabel('Angle RMSE (degree)');
title([str_indp_cohr ', M=' num2str(M) ', N=' num2str(N)]);

 legend();
%%
if Flag_save_fig
    saveas(h2, [results_dir '/MSE.fig' ]);
    exportgraphics(gcf,[results_dir '/MSE.jpg']);
end




