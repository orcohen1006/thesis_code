clear; close all; clc; %restoredefaultpath;
rng(42);
% pool = gcp('nocreate');
% if isempty(pool), parpool("Processes"); end
%%
Large_Scale_Flag = false;
t0_overall = tic;
%%
% for cohr_flag = [false, true]
%     for N = [16,120]
%         Exp_SNR(N, cohr_flag, Large_Scale_Flag);
%     end
% end
%%
for cohr_flag = [false, true]
    for N = [16,120]
        Exp_DeltaTheta(N, cohr_flag, Large_Scale_Flag);
    end
end
%%
% for cohr_flag = [false, true]
%     Exp_N(cohr_flag, Large_Scale_Flag);
% end
%%
disp(['Total Running Time at M-file: ' num2str(toc(t0_overall)) ' sec.']);













