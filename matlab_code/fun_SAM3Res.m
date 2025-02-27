%% =============== SAM-3 estimates Results ==========================
function [Detected_powers, Distance, p_vec, normal, noisepower]=fun_SAM3Res(Y,A,DAS_init,DOAscan,DOA, sigma_given)
% ---------------------------------------------------
% output list: 
% Distance 1 x # source, row vector
% p_vec: # scan point x 1, col vector
% normal: tag, 
% if normal == 1, detecion is Okay
% otherwise normal ==0, detectio failed
%
%
% Input list: 
% Y: measured data, each col. is one snapshot
% A: steering vector matrix
% DAS_init: intitial coefficients estimates by DAS
% DOAscan: grid
% DOA: truth
% Sep. 4, 2011 by QL
% ---------------------------------------------------
flag_sigma_is_given = exist("sigma_given","var") && ~isempty(sigma_given);

Numsources =length(DOA);
threshold = 1e-4;
maxIter= 1e4;30;
% colorSet={'r-', 'b-', 'r-.', 'b-.', 'r--', 'b-.', 'r:', 'b:'};

[M thetaNum]=size(A);
t_samples = size(Y, 2);
R_N = (Y*Y')/t_samples;
if flag_sigma_is_given
    sigma = sigma_given;
else
    sigma = mean(abs(Y(:)).^2);
end
p_vec_Old = abs(DAS_init).^2; % 
% coeffOld = DAS_init;

for iterIdx = 1:maxIter
    R =  A*spdiags(p_vec_Old, 0, thetaNum, thetaNum)*A' + sigma*eye(M);

    Rinv=inv(R);
    Rinv_A = Rinv*A;
    diag_A_Rinv_A = sum(conj(A).*Rinv_A, 1).';
    tmp = sum(conj(A).* (Rinv*R_N*Rinv*A), 1).';
	p_vec = p_vec_Old.*(tmp./diag_A_Rinv_A);
    
    
%     p_vec = p_vec_Old.*(diag(A'*Rinv*R_N*Rinv*A)./diag_A_Rinv_A);
    if ~flag_sigma_is_given
        sigma = real(trace(Rinv*Rinv*R_N))/real(trace(Rinv*Rinv));
    end
    p_diffs_ratio = norm(p_vec_Old-p_vec)/norm(p_vec_Old);
    if p_diffs_ratio < threshold
%         disp(['=== SAM-3 convgs at iteration ' num2str(iterIdx)]);
%         disp('p_vec change ratio == ');
%         disp(p_diffs_ratio);
        break;
    end
    p_vec_Old = p_vec;
end

p_vec = real(p_vec);
[pks index]=findpeaks(p_vec, 'sortstr', 'descend');

if length(index) < Numsources
%     warning('Not all peaks detected');
    normal = 0;
    Distance = NaN;
%     p_vec = NaN;
    Detected_powers = NaN;
    noisepower = sigma;
    return;
end

% ------------ Check whether the detection is right -----
Detected_DOAs = DOAscan(index(1:Numsources));

[Detected_DOAs, IXsort] = sort(Detected_DOAs, 'ascend');
Distance = Detected_DOAs - DOA;
% if max(abs(Distance)) > 10
% %     warning('Failed Detection by SAM-3, this simulation is abnormal!');
%     normal = 0;
%     Distance = NaN;
% %     p_vec = NaN;
%     Detected_powers = NaN;
% else
%     normal = 1; % detection okay
%     % the powers from large value to small value
%     Detected_powers = p_vec(index(1:Numsources));
%     % sort the power according to the DOA 
%     Detected_powers =  Detected_powers(IXsort);
% end

    normal = 1; % detection okay
    % the powers from large value to small value
    Detected_powers = p_vec(index(1:Numsources));
    % sort the power according to the DOA 
    Detected_powers =  Detected_powers(IXsort);



noisepower = sigma;




end
