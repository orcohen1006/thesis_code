%% =============== DAS estimates ==========================
function [Detected_powers,Distance, p_vec, normal, noisepower]=fun_DASRes(Y,A,DAS_init,DOAscan,DOA)
% Put the private function out 
% Updated Jun 19, 2011 QL
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
%  Updated Aug 21, 2011 QL
% ---------------------------------------------------
noisepower = NaN; % not able to give it
[M thetaNum]=size(A);
t_samples = size(Y, 2);
modulus_hat_das  = sum(abs(A'*Y/M), 2 )/t_samples;
p_vec = abs(modulus_hat_das).^2;

Numsources =length(DOA);
[pks index]=findpeaks(p_vec, 'sortstr', 'descend');

if length(index) < Numsources
%     waring('Not all peaks detected');
    normal = 0;
    Distance = NaN;
%     p_vec = NaN;
    Detected_powers = NaN;
    return;
end

% ------------ Check whether the detection is right -----
Detected_DOAs = DOAscan(index(1:Numsources));

[Detected_DOAs, IXsort]= sort(Detected_DOAs, 'ascend');
Distance = Detected_DOAs - DOA;

% if max(abs(Distance)) > 20
% %     warning('Failed Detection by DAS, this simulation is abnormal!');
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


end