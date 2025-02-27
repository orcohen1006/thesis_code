%% =============== SPICE+ estimates ==========================
function [Detected_powers, Distance, p, normal, noisepower]=fun_SPICE_fast(Y,A,DAS_init,DOAscan,DOA, sigma_given)
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
% Sep. 5, 2011 QL
% ---------------------------------------------------
flag_sigma_is_given = exist("sigma_given","var") && ~isempty(sigma_given);
Numsources =length(DOA);

maxIter=1e4;25;
% colorSet={'r-', 'b-', 'r-.', 'b-.', 'r--', 'b-.', 'r:', 'b:'};
EPS_NORM_CHANGE = 1e-4;

[M N]=size(A);
t_samples = size(Y, 2);
R_hat = (Y*Y')/t_samples;
%sigma = mean(abs(Y(:)).^2);%;1e-6;
p = abs(DAS_init).^2; % 
S = sort(p, 'ascend');

sigmainit = mean(S(1:M)); % estimate SigmaInit

% ==========================================
% make R_hat toeplitz using sample mean
%R_hat_row = zeros( 1, M);
%for setvalue = 1:M
%    R_hat_row(setvalue) = mean(diag(R_hat, setvalue -1 ));    
%end
%R_hat = toeplitz(R_hat_row); % modified R_hat, force toeplitz constraint
invR_hat_A = R_hat\A;
% --------------------------------------

% Enhanced SPICE, SPICE+, with identical \sigma_k
% ======== compute weight using conventional way =====
weight_first = real(sum(conj(A).*invR_hat_A,1))/M;

weight        = [ weight_first , diag(inv(R_hat)).'/M ];  % modi
gamma         = mean(diag(inv(R_hat))); % modi
if flag_sigma_is_given
    sigma = sigma_given;
else
    sigma         = sigmainit; % using SPCIE+, with equal sigma constraint
end


for jj        = 1:maxIter
%     display([' Slow SPICE+ Iteration ',num2str(jj),'...']);
    p_prev = p;
    % ----------- Prepare ------------------------------------
    P         = diag(p);
    R         = A * P * A' + sigma * eye(M);
    Rinv_R_hat_sqrt = R\sqrtm(R_hat);
    
    
    % ------------ compute rho ----------------------------
%     rho =0;
%     am_Rinv_R_hat_sqrt = zeros(N , M); 
%     norm_am_Rinv_R_hat_sqrt = zeros(N , 1); % save for future use
%     for idx = 1: N
%         am_Rinv_R_hat_sqrt(idx,:) = A(:,idx)' * Rinv_R_hat_sqrt; 
%         norm_am_Rinv_R_hat_sqrt(idx) = sqrt(sum(diag(am_Rinv_R_hat_sqrt(idx,:) * am_Rinv_R_hat_sqrt(idx,:)' )));
%         rho = rho + sqrt(weight(idx)) * p(idx) * norm_am_Rinv_R_hat_sqrt(idx);
%     end

    % <!! Or's code optimization !!>
    am_Rinv_R_hat_sqrt = A'* Rinv_R_hat_sqrt;
    norm_am_Rinv_R_hat_sqrt = vecnorm(am_Rinv_R_hat_sqrt,2,2);
    rho = sum(sqrt(weight(1:N)') .* p .* norm_am_Rinv_R_hat_sqrt);
    % <!! Or's code optimization !!>
    
    % keep the  || ||F for future use
    norm_Rinv_Rhatsqrt = sqrt(sum( diag( Rinv_R_hat_sqrt' * Rinv_R_hat_sqrt ) ) );  % save for future use
    rho = rho + sqrt(gamma) * sigma * norm_Rinv_Rhatsqrt;
    
    
    % ------------- compute sigma ---------------------------------
    if ~flag_sigma_is_given
        sigma = sigma * norm_Rinv_Rhatsqrt / (sqrt(gamma) * rho  );
    end
   
    
    % --------------- compute p ------------------------------------
%     for pidx = 1: N
%         p(pidx) = tmp(pidx) * norm_am_Rinv_R_hat_sqrt(pidx)  / (rho * sqrt(weight(pidx) ) );
%     end

    % <!! Or's code optimization !!>
    p = p .* norm_am_Rinv_R_hat_sqrt ./ (rho * sqrt(weight(1:N)'));
    % <!! Or's code optimization !!>
    
    
    
    p = abs(p);

    measured_change_norm = norm(p -p_prev)/norm(p);
    if measured_change_norm < EPS_NORM_CHANGE
        break;
    end
end

[pks index]=findpeaks(p, 'sortstr', 'descend');

if length(index) < Numsources
%     warning('Not all peaks detected');
    normal = 0;
    Distance = NaN;
%     p = NaN;
    Detected_powers = NaN;
    noisepower = sigma;
    return;
end

% ------------ Check whether the detection is right -----
Detected_DOAs = DOAscan(index(1:Numsources));

[Detected_DOAs, IXsort] = sort(Detected_DOAs, 'ascend');
Distance = Detected_DOAs - DOA;
% if max(abs(Distance)) > 10
% %     warning('Failed Detection by SPICE, this simulation is abnormal!');
%     normal = 0;
%     Distance = NaN;
% %     p = NaN;
%     Detected_powers = NaN;
% else
%     normal = 1; % detection okay
%     % the powers from large value to small value
%     Detected_powers = p(index(1:Numsources));
%     % sort the power according to the DOA 
%     Detected_powers =  Detected_powers(IXsort);
% end


    normal = 1; % detection okay
    % the powers from large value to small value
    Detected_powers = p(index(1:Numsources));
    % sort the power according to the DOA 
    Detected_powers =  Detected_powers(IXsort);



noisepower = sigma;



end
