function Exp_N(cohr_flag, Large_Scale_Flag)
rng(42);
Flag_save_fig = true;
M = 12;
timestamp = datestr(now, 'dd-mm-yyyy_HH-MM-SS');
str_indp_cohr = 'indp';
if (cohr_flag), str_indp_cohr = 'cohr'; end
results_dir = ['Exp_N_' timestamp '_' str_indp_cohr '_M' num2str(M) '_L' num2str(Large_Scale_Flag) '/' ];



if ~exist(results_dir, 'dir') && Flag_save_fig
    mkdir(results_dir);
end


if Large_Scale_Flag
    disp('==== Large SCALE MC tests Running...');
    NUM_MC =  500;
    VEC_N= [16, 30, 80, 120, 500];
else
    disp('=========== SMALL SCALE MC tests@@@ !!! =======');
    NUM_MC =  50;
    VEC_N= [16, 30, 80, 120, 500];
end
SNR = 0;

algo_list = ["PER", "SPICE", "SAMV", "AFFINV"];

num_algos = length(algo_list);

SE_mean = zeros(length(VEC_N), num_algos);
Failing_rate = zeros(length(VEC_N), num_algos);


CRBlist = zeros(1, length(VEC_N));

%%
PowerDOAdB = [3;4];%[5; 3]; % in dB
DOA = [35.11 50.15]; % true DOA angles, off gird case
%%
for N_ind = 1:length(VEC_N)
    N = VEC_N(N_ind);
    disp(['=== Computing N == ' num2str(N)]);
    
    [SE_mean_perAlgo, Failing_rate_perAlgo, CRB_val] = ...
        ComputeAlgosStdErr(algo_list, NUM_MC, SNR, N, M, cohr_flag, PowerDOAdB, DOA);
    
    SE_mean(N_ind, :) = SE_mean_perAlgo;
    Failing_rate(N_ind,:) = Failing_rate_perAlgo;
    CRBlist(N_ind) = CRB_val;
    
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
    semilogy(VEC_N, prepare, colorSet{i_algo},'DisplayName',algo_list(i_algo));
    hold on;
end

% -----CRB ----
semilogy(VEC_N, sqrt(CRBlist), colorSet{num_algos+1},'DisplayName','CRB'); 

xlabel('N','Interpreter','latex');
ylabel('Angle RMSE (degree)');
title([str_indp_cohr ', M=' num2str(M) ', N=' num2str(N)]);

 legend();
%%
if Flag_save_fig
    saveas(h2, [results_dir '/MSE.fig' ]);
    exportgraphics(gcf,[results_dir '/MSE.jpg']);
end




