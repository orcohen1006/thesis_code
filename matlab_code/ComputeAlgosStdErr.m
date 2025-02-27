function [SE_mean_perAlgo, Failing_rate_perAlgo, CRB_val] = ...
    ComputeAlgosStdErr(algo_list, NUM_MC, SNR, t_samples, M, cohr_flag, PowerDOAdB, DOA)
%%
num_algos = length(algo_list);
Failed_total_times = zeros(num_algos, 1);
SE_history = zeros(num_algos, NUM_MC);
%%
% Fixed Source powers
num_sources = length(DOA); % # of sources

PowerDOA = 10.^(PowerDOAdB/10);
amplitudeDOA = sqrt(PowerDOA);

DOAscan = 0: 0.5 :180; % doa grid

DOA = sort(DOA, 'ascend');

delta_vec = 0:(M-1);
A_true = exp(1j*pi*delta_vec' * cos(DOA*pi/180) ); % true steering vector matrix
A = exp(1j*pi*delta_vec' * cos(DOAscan*pi/180) ); % steering vector matrix w.r.t all possible scanning DOA's

noisePowerdB = mean(PowerDOAdB(:)) - SNR;
noisePower = 10^(noisePowerdB /10);
%%
for i_MC = 1:NUM_MC %%%%%%%%%%%%%% parfor
    t0 = tic;
    %% ========================= Generate signal
    noise = sqrt(noisePower) * (randn(M,t_samples) + 1j* randn(M, t_samples))/sqrt(2);
    if ~cohr_flag % indp sources
        waveform = exp(1j*2*pi*rand(num_sources, t_samples)) .* repmat(amplitudeDOA, 1, t_samples);
    else % cohr sources
        waveform = exp(1j*2*pi*rand(num_sources-1, t_samples));
        waveform = [waveform;  waveform(1, :)  ];
        waveform = waveform.* repmat(amplitudeDOA , 1, t_samples);
    end

    y_noisefree = A_true *  waveform; % ideal noiseless measurements
    y_noisy      = y_noisefree + noise; % noisy measurements

    modulus_hat_das  = sum(abs(A'*y_noisy/M), 2 )/t_samples;
    %% ========================= Run on all algorithms
    SqrErr = cell(num_algos, 1);
    PowerSE = cell(num_algos, 1);
    p_vec_cell = cell(num_algos, 1);
    for i_algo = 1:num_algos
        Detected_powers = nan;
        Distance = nan;
        normal = nan;
        switch algo_list(i_algo)
            case "PER"
                [Detected_powers, Distance, p_vec, normal, ~]=fun_DASRes(y_noisy, A, modulus_hat_das,DOAscan,DOA);
            case "SAMV"
                [Detected_powers, Distance, p_vec, normal, ~] = fun_SAM3Res(y_noisy, A, modulus_hat_das,DOAscan,DOA, noisePower);
            case "SPICE"
                [Detected_powers, Distance, p_vec, normal, ~] = fun_SPICE_fast(y_noisy, A, modulus_hat_das,DOAscan,DOA,noisePower);
            case "AFFINV"
                [Detected_powers, Distance, p_vec, normal, ~] = fun_Affinv(y_noisy, A, modulus_hat_das,DOAscan,DOA, noisePower);
            otherwise
                error("not implemented");
        end
        p_vec_cell{i_algo} = p_vec;
        if ~normal
            SqrErr{i_algo} = NaN;
            PowerSE{i_algo} = NaN;
        else
            power_dif = Detected_powers - PowerDOA;
            SqrErr{i_algo} = Distance * Distance';
            PowerSE{i_algo} = power_dif.' * power_dif;
        end
    end
    %%
    if (false)
        %%
        figure; grid on; hold on;
        plts = [];
        for i_algo = 1:num_algos
            plt = plot(DOAscan, 10*log10(p_vec_cell{i_algo}),'-o','DisplayName',algo_list(i_algo));
            plts = [plts, plt];
        end
        plt = plot(DOA,PowerDOAdB,'x','DisplayName','DOA');
        plts = [plts, plt];
        legend(plts);
    end
    %% =========================
    SE_all_m = cell2mat(SqrErr);
    % test each algorithm
    NaN_flag_col_vec = isnan(SE_all_m); % use the flag to do the magic
    Failed_total_times = Failed_total_times + NaN_flag_col_vec;


    SE_all_m(NaN_flag_col_vec) = inf; % change NaN to inf
    SE_history(:, i_MC) = SE_all_m;
    %         SE_sum  = SE_sum   + SE_all_m;
    dt = toc(t0);
    disp("i_MC = " + i_MC + ", elapsed time: " + dt + "[sec]");

end % MC
%% Prepare output
SE_history_sorted = sort(SE_history, 2, 'descend');

% discard 2% bad results
percent = 2;
actual_del_entries = ceil(NUM_MC * percent/100);

SE_mean_perAlgo = zeros(1, num_algos);
Failing_rate_perAlgo = zeros(1, num_algos);
for i_algo = 1: num_algos
    start_pointer = find(SE_history_sorted(i_algo, :) < inf, 1, 'first' );
    count_pointer = start_pointer + actual_del_entries;
    SE_mean_perAlgo(i_algo) = mean(  SE_history_sorted(i_algo, count_pointer:end) );
    Failing_rate_perAlgo(i_algo) = Failed_total_times(i_algo) /NUM_MC;
end


CRB_val = SAM_CRB(SNR, t_samples, cohr_flag, PowerDOAdB, DOA);
%%
end













