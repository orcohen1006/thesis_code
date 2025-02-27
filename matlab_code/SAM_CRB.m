function [mse_CBRDOA] = SAM_CRB(SNR_in, t_sample_in, cohr_flag, PowerDOAdB, DOA)  
% Sep. 1, 2011 by QL
% simple CRB for the SAMV paper draft
% using the stochastic CRB definitions...

M  = 12; 
t_samples = t_sample_in; % 120


% Fixed Source powers
if ~exist("PowerDOAdB", "var")
    PowerDOAdB = [3;4];%[5; 3]; % in dB
end
PowerDOA = 10.^(PowerDOAdB/10);
amplitudeDOA = sqrt(PowerDOA);

% complex Gaussian Noise
SNR = SNR_in; % input parameters
noisePowerdB = mean(PowerDOAdB(:)) - SNR; 
 
noisePower = 10^(noisePowerdB /10);
 


Dist = ones(1, M-1); % inter-element spacing of sensors
DistTmp = cumsum([0 Dist]); % locations of the M sensors


if ~exist("DOA", "var")
    DOA = [35.11 50.15]; % true DOA angles, off gird case
end
DOA = sort(DOA, 'ascend'); % must be in accend order to work right
 



source_no = length(DOA); % # of sources


A = exp(1j*pi*DistTmp' * cos(DOA*pi/180) ); % real steering vector matrix
DA = (-1j*pi*DistTmp' * sin(DOA*pi/180) *pi/180  )   .*  A; 

PI_A = eye(M) - A* (( A' * A ) \ (A'));

% compute P, dependent on the cohr_flag
if ~cohr_flag % indp sources
    Pmat = diag(PowerDOA);
else
    rho = 1; % 100% coherence assumption
    % for simplicity, assuming only 2 sources
    Pmat = [  PowerDOA(1), rho * amplitudeDOA(1)* amplitudeDOA(2); ...
              rho * amplitudeDOA(2)* amplitudeDOA(1), PowerDOA(2)];
end
R = A * Pmat * A' + noisePower * eye(M);
Inside = (DA' * PI_A * DA ) .* ( (Pmat * A' * inv(R) * A * Pmat ).' );  
CRB_matrix =  inv(real(Inside) ) * noisePower / (2 * t_samples);
mse_CBRDOA = trace(CRB_matrix);


end % end function 






