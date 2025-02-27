function Parfor_MC_SAMS_runDeltaTheta(snap_value, M_value, cohr_flag, ...
    Flag_save_fig, Large_Scale_Flag)
% filename: Parfor_MC_SAMS.m

% Updated sliced indexing...

% Parfor and matlabpools...  
% Aug 22, 2011 by QL....

% % for debug only...
% disp('========== Debugging mode at Function: Parfor_MC_SAMS.m !!! ');
% snap_value = 16;
% M_value = 12;
% cohr_flag = 1;
% Flag_save_fig = 0;
% Large_Scale_Flag = 0;
% % end for debug only...


% ***************
% CoreNum = 8; % this is the number of cores in the running workstation.
% % disp(['**** Assumed Num of Cores in the Workstation ===  ' num2str(CoreNum)   ]);
% % disp('===== Trying ... Starting Parallel MATLAB clients sessions =====');
% currentClients_no = matlabpool('size');
% if abs(currentClients_no - CoreNum)  > eps % already a matlabpool working
%     disp(['--- Already a matlabpool running but not equal to Num of COREs' num2str(CoreNum) ', trying to restart matlabpool...']);
%     matlabpool close;
%     matlabpool('open', CoreNum);
% end
% % no matlabpool running ......


timestamp = datestr(now, 'dd-mm-yyyy_HH-MM-SS');

if ~cohr_flag
    figpath = ['SAM_ML_DELTA_THETA_' timestamp '/ParFor_indp_M' num2str(M_value) '_N' num2str(snap_value) '_L' num2str(Large_Scale_Flag) '/' ];
else 
    figpath = ['SAM_ML_DELTA_THETA_' timestamp '/ParFor_cohr_M' num2str(M_value) '_N' num2str(snap_value) '_L' num2str(Large_Scale_Flag) '/' ];
end


if ~exist(figpath, 'dir') && Flag_save_fig
    mkdir(figpath);
end


if Large_Scale_Flag
    error("what?");
else
    disp('=========== SMALL SCALE MC tests@@@ !!! =======');
    Total_Monte_Tials =  200;
    SNR = 0;
    VEC_DELTA_THETA= 2:10;
    % SNRSetdB= [-15 -10 -5 0 10]; %[10]; % sometimes the -10 SNR would produe errors
end


algo_list = {'PER', 'IAA', 'SAMV1', 'SAMV2', 'SPICE+', 'AMV-SML', 'SAMV1-SML', 'SAMV2-SML', 'MUSIC','Affinv'};
ids_selected_algos = [1,4,5,10];

% algo_list = {'PER', 'IAA'};
Num_algos = length(algo_list); % 

SE_mean = zeros(length(VEC_DELTA_THETA), Num_algos);
Failing_rate = zeros(length(VEC_DELTA_THETA), Num_algos);


CRBlist = zeros(1, length(VEC_DELTA_THETA)); % row vec

PowerDOAdB = [3;4];%[5; 3]; % in dB
% DOA = [35.11 50.15]; % true DOA angles, off gird case
firstDOA = 35.11;
for delta_theta_ind = 1:length(VEC_DELTA_THETA)
    delta_theta = VEC_DELTA_THETA(delta_theta_ind);
    disp(['=== Computing DeltaTheta == ' num2str(delta_theta)]);
    DOA = [firstDOA, firstDOA + delta_theta];
    % zero out the summation temp variables for all algorithms
    Failed_total_times = zeros(Num_algos, 1);
    SE_history = zeros(Num_algos, Total_Monte_Tials);
    

    
    % 1 MC trial
    parfor idx_MC = 1: Total_Monte_Tials %%%%%%%%%%%%%% parfor
        t0 = tic;
%     for idx_MC = 1: Total_Monte_Tials
        [SE_all,  ~] = ...
            Compute_algos_StdErr(SNR, snap_value, M_value, cohr_flag, PowerDOAdB, DOA);

        SE_all_m = cell2mat(SE_all);

        
        
        % test each algorithm
        NaN_flag_col_vec = isnan(SE_all_m); % use the flag to do the magic
        Failed_total_times = Failed_total_times + NaN_flag_col_vec;
            
            
        SE_all_m(NaN_flag_col_vec) = inf; % change NaN to inf
        SE_history(:, idx_MC) = SE_all_m;
%         SE_sum  = SE_sum   + SE_all_m;
        dt = toc(t0);
        disp("idx_MC = " + idx_MC + ", elapsed time: " + dt + "[sec]");
    end % end MC trials
    SE_history_sorted = sort(SE_history, 2, 'descend');
    
    
    % discard 2% or 5%(for example ) bad results
    percent = 2;
    actual_del_entries = ceil(Total_Monte_Tials * percent/100);
    
    
    for idx = 1: Num_algos
        start_pointer = find(SE_history_sorted(idx, :) < inf, 1, 'first' );
        count_pointer = start_pointer + actual_del_entries;
        
        SE_mean(delta_theta_ind, idx) = mean(  SE_history_sorted(idx, count_pointer:end) );
        
        
        Failing_rate(delta_theta_ind, idx) = Failed_total_times(idx) /Total_Monte_Tials;
    end
    
    
    
   

    % ---- CRLB: this is correct only with M = 12. 
    CRBlist(delta_theta_ind) = SAM_CRB(SNR, snap_value, cohr_flag,PowerDOAdB, DOA);
    
end % end for SNR....




if Flag_save_fig
    save([figpath '/Algos_Data.mat' ]);%, 'SNR','VEC_DELTA_THETA', 'SE_mean', 'Failing_rate', 'algo_list' );
end

%% ==== Plot figures =========

% % old_colorSet={'r-', 'b-', 'k-.', 'c-.', 'r--', 'b-.', 'r:', 'b:'};
% % old_colorSet={'r-->', 'k-s', 'b--*', 'm--x', 'r-o', 'k--d', 'b-^', 'm-p'};
% updated color-scheme
colorSet={'r-->', 'c--s', 'k--+', 'b-^', 'm--p', 'b--*',  'r-o',  'm-x', 'c-v','g--s', 'k-d'};
% colorset = colorset(ids_selected_algos);

%{
%% Use fixed failure rate assumption, hence the failure rate figure is
%% deleted here.
% plot failing rate figures
h1 = figure; 
for ind = 1: Num_algos
    plot(SNRSetdB, Failing_rate(:,ind).', colorSet{ind} );
    hold on;
end
xlabel('SNR (dB)');
ylabel('Failure Probability');
if ~cohr_flag
    title(['Indpendent M=' num2str(M_value) ', N=' num2str(snap_value)]);
else
    title(['Coherent M=' num2str(M_value) ', N=' num2str(snap_value)]);
end
legend(algo_list, 0);
myboldify1;
if Flag_save_fig
    saveas(h1, [figpath '/failingrate.fig' ]);
    exportgraphics(gcf,[figpath '/failingrate.jpg']);
end
%}



% plot SE figure;
h2 = figure;
for ind = ids_selected_algos %1: Num_algos
    prepare = SE_mean(:,ind).' + eps;
    prepare = sqrt(prepare);
    % prepare = prepare ./ VEC_DELTA_THETA;
    semilogy(VEC_DELTA_THETA, prepare, colorSet{ind},'DisplayName',algo_list{ind});
    hold on;
end

% -----CRB ----
semilogy(VEC_DELTA_THETA, sqrt(CRBlist), colorSet{ind+1},'DisplayName','CRB'); 
% semilogy(VEC_DELTA_THETA, sqrt(CRBlist)./VEC_DELTA_THETA, colorSet{ind+1},'DisplayName','CRB'); 
% lgd_list = [algo_list, 'CRB'];

xlabel('$\Delta \theta$ (degrees)','Interpreter','latex');
ylabel('Angle RMSE (degree)');
if ~cohr_flag
    title(['Indpendent M=' num2str(M_value) ', N=' num2str(snap_value)]);
else
    title(['Coherent M=' num2str(M_value) ', N=' num2str(snap_value)]);
end
 
% legend(lgd_list);
legend();
%%
% myboldify1;
if Flag_save_fig
    saveas(h2, [figpath '/MSE.fig' ]);
    exportgraphics(gcf,[figpath '/MSE.jpg']);
end


%{
%% Discared the power MSE figure in the paper... 
% plot Power MSE figure;
h3 = figure;
for ind = 1: Num_algos 
    Prepare_plot = PowerErr_mean(:,ind).';
    semilogy(SNRSetdB, Prepare_plot +eps , colorSet{ind});
    hold on;
end
xlabel('SNR (dB)');
ylabel('Power MSE');
if ~cohr_flag
    title(['Indpendent M=' num2str(M_value) ', N=' num2str(snap_value)]);
else
    title(['Coherent M=' num2str(M_value) ', N=' num2str(snap_value)]);
end

% MUSIC, the one without power estimates should always comes in the last...
if strcmpi(algo_list{end}, 'MUSIC')
    disp('Right Order, deleting useless MUSIC power entry');
    algo_list(end) = []; % delete the extra MUSIC entry while plotting the power figure
else
    warning('Possibly Wrong Algorithm orders!!');
end


legend(algo_list, 0);
myboldify1;

if Flag_save_fig
    saveas(h3, [figpath '/PowerSE.fig' ]);
    exportgraphics(gcf,[figpath '/PowerSE.jpg']);
end
%}








