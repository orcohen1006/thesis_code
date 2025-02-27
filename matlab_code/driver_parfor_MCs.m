% driver_parfor_MCs.m
% remember to close all after each MC_calls...
%
% calls: MC_algos_run...
%     (snap_value, M_value, cohr_flag, Flag_save_fig, Large_Scale_Flag)
%
% updated Sep. 4, 2011 by QL 
clear; close all; clc;
addpath('..');
debug_flag = 1; % 1; % if 1, debugging mode on, only computes very few algorithms
% CoreNum = 8;
% disp(['**** Assumed Num of Cores in the Workstation ===  ' num2str(CoreNum)   ]);
% disp('===== Trying ... Starting Parallel MATLAB clients sessions =====');
% currentClients_no = matlabpool('size');
% if abs(currentClients_no)  > eps % already a matlabpool working
%     disp(['--- Already a matlabpool running, trying to restart matlabpool...']);
%     matlabpool close;
% end
% matlabpool('open', CoreNum);
%%
pool = gcp('nocreate');
if isempty(pool), parpool("Processes"); end
%%

if ~debug_flag % not debugging
    tic;
    disp('---Testing M = 12----');
    for N = [16 120]
        disp(['>> M = 12, N = ' num2str(N) ' Indp: ' ]);
        Parfor_MC_SAMS(N, 12, 0, 1, 0);
        close all;

        disp(['>> M = 12, N = ' num2str(N) ' Cohr: ' ]);
        Parfor_MC_SAMS(N, 12, 1, 1, 0);
        close all;


    end
    t_M12 = toc;

else
% -------- debuggginng...... ----------
    tic;
    disp('---Debugging... Testing M = 12----');
    M = 12;
    cohr_flag = false;
    %%
    % for N = [16  120]
    %     disp(['>> M = 12, N = ' num2str(N) ' Indp: ' ]);
    % 
    %     % Parfor_MC_SAMS(N, M, cohr_flag, 1, 0);
    %     Parfor_MC_SAMS_runDeltaTheta(N, M, cohr_flag, 1, 0);
    % end
    %%
    Parfor_MC_SAMS_runN([], M, cohr_flag, 1, 0);
    %%
    t_M12 = toc;
    % -------- End debugging...... ---------
end

% if matlabpool('size') > eps % 
%     disp('== parallel finished! Closing clients...');
%     matlabpool close;
% end

disp(['Total Running Time at M-file: driver_parfor_MC: ' num2str(t_M12) ' sec.']);













