%% gen teste dataset
% Script to generate M received training symbols for every MIMO channel in the matrix obtained from NYUSIM channel simulator
% - A fully connected hybrid MIMO architecture is assumed at the TX and RX. 
% - The received training pilot is obtained at the output of the ADCs for the different RF chains. 
% - Only analog precoding/combining during training is used. 
% - ULAs are assumed
% References:
% [1] J. Rodr�guez-Fern�ndez, N. Gonz�lez-Prelcic, K. Venugopal and R. W. Heath, "Frequency-Domain Compressive Channel Estimation for Frequency-Selective Hybrid Millimeter Wave MIMO Systems," IEEE Transactions on Wireless Communications, vol. 17, no. 5, pp. 2946-2960, May 2018.

function [CH, TH] = ReconstructChannel_train(Pilots, Phi, N_RX, N_TX, D_w)
    % function R = ReconstructChannel(Pilots, Phi, N_RX, N_TX)
    %
    % INPUTS
    % Pilots - Measured antenna configurations with dimensions (N_Phi, N_fft)
    % Phi - Used antenna configurations with dimensions (N_Phi, N_RX*N_TX)
    % N_RX - Number of antenna elements in the receiver
    % N_TX - Number of antenna elements in the transmitter
    % Note: N_Phi is the number of used antenna configurations and N_fft the
    % number of subcarriers
    %
    % OUTPUTS
    % Ch - Reconstructed channel
    % AoA - Angles of arrival for the computed paths
    % AoD - Angles of departure for the computed paths
    % ToF - Times of flight for the computed paths
    % Alpha - Complex gains for the computed paths
    %% Parameters
    Time_sr_d = 4;             % Time sampling ratio
    Time_sr_r = 32;            % Time sampling ratio for refinement
    Angle_sr_d = 2;            % Angle sampling ratio for detection
    Angle_sr_r = 32;           % Angle sampling ratio for refinement
    Confidence_min = 0.7;      % Detection confidence minimum
    PingPong_it = 3;           % Number of iteration to refine the angles
    bool_debug = false;        % Debug condition for plots
    %% Whitening
    if nargin > 4
        Pilots = D_w\Pilots;
        Phi = D_w\Phi;
    end
    %% Compute N_Phi and N_fft
    [N_Phi, N_fft] = size(Pilots);
    %% Derived parameters
    Time_res_d = Time_sr_d*N_fft;
    Time_res_r = Time_sr_r*N_fft;
    %% Compute angle responses
    Angle_RX_r = linspace(-pi, pi, N_RX*Angle_sr_r+1); Angle_RX_r(end) = [];
    Angle_TX_r = linspace(-pi, pi, N_TX*Angle_sr_r+1); Angle_TX_r(end) = [];
    A_RX_r = exp((0:N_RX-1).'*Angle_RX_r*1i)/sqrt(N_RX);
    A_TX_r = exp((0:N_TX-1).'*Angle_TX_r*1i)/sqrt(N_TX);
    %% Compute simple angle transform
    if Angle_sr_d < 2
        Angle_d = linspace(-pi, pi, N_RX*Angle_sr_d+1); Angle_d(end) = [];
        A_RX_d = SincBeam(N_RX, 1.1*2*pi/(N_RX*Angle_sr_d)).*exp((0:N_RX-1).'*Angle_d*1i);
        Angle_d = linspace(-pi, pi, N_TX*Angle_sr_d+1); Angle_d(end) = [];
        A_TX_d = SincBeam(N_TX, 1.1*2*pi/(N_TX*Angle_sr_d)).*exp((0:N_TX-1).'*Angle_d*1i);
    else
        Angle_d = linspace(-pi, pi, N_RX*Angle_sr_d+1); Angle_d(end) = [];
        A_RX_d = exp((0:N_RX-1).'*Angle_d*1i);
        Angle_d = linspace(-pi, pi, N_TX*Angle_sr_d+1); Angle_d(end) = [];
        A_TX_d = exp((0:N_TX-1).'*Angle_d*1i);
    end
    A_RTX_d = kron(A_TX_d.', A_RX_d');
    %% Dump pilots info into the residual pilots and convert it to time information
    Pilots_res = Pilots;
    Pilots_res_t = ifft(Pilots_res, Time_res_d, 2);
    Pilots_res_t_esp = (A_RTX_d*Phi')*Pilots_res_t;
    %% Compute noise level and the confidence threshold
    Noise_level = sqrt(1/(2*log(2)))*median(abs(Pilots_res_t_esp(:)), 1);
    Confidence_TH = Noise_level*sqrt(-2*log(1-Confidence_min^(1/numel(Pilots_res_t_esp))));
    %% Plot noise level and std margins
    if bool_debug
        figure(1)
        plot(max(abs(Pilots_res_t_esp)).', 'b', 'LineWidth', 1.5), hold on
        plot([1, Time_res_d], repmat(Confidence_TH, 1, 2), 'r', 'LineWidth', 1);hold off
    end
    %% Initialization
    CH = cell(0);
    TH = [];
    %% Path substraction loop
    AoA = [];
    AoD = [];
    ToF = [];
    H   = [];
    P   = [];
    Alpha = [];
    [amax, imax] = max(abs(Pilots_res_t_esp(:)));
    while amax > Confidence_TH
        %% Imaging
        [ii_aoa, ii_aod, ii_t] = ind2sub([N_RX*Angle_sr_d, N_TX*Angle_sr_d, Time_res_d], imax);
        %% Extract time and spatial measurements
        t = (ii_t-1)/Time_res_d; % Time computed between 0 and 1
        Mt = Pilots_res_t(:, ii_t);
        %% Pseudochannel computation
        % These two options are similar but have different properties
        H_pseudo = reshape(Phi'*Mt, [N_RX, N_TX]);
    %     H_pseudo = reshape(Phi\Mt, [N_RX, N_TX]);
        %% First iteration estimation
        if N_RX > N_TX
            [~, ii_aoa] = max(abs((H_pseudo*A_TX_d(:, ii_aod))'*A_RX_r));
            [~, ii_aod] = max(abs((A_RX_r(:, ii_aoa)'*H_pseudo)*A_TX_r));
        else
            [~, ii_aod] = max(abs((A_RX_d(:, ii_aoa)'*H_pseudo)*A_TX_r));
            [~, ii_aoa] = max(abs((H_pseudo*A_TX_r(:, ii_aod))'*A_RX_r));
        end
        [~, ii_t] = max(abs(ifft((kron(A_TX_r(:, ii_aod)', A_RX_r(:, ii_aoa).')*Phi')*Pilots_res, Time_res_r, 2)));
        %% Ping pong
        for iter = 1:PingPong_it
            if N_RX > N_TX
                [~, ii_aoa] = max(abs((H_pseudo*A_TX_r(:, ii_aod))'*A_RX_r));
                [~, ii_aod] = max(abs(A_RX_r(:, ii_aoa)'*H_pseudo*A_TX_r));
            else
                [~, ii_aod] = max(abs(A_RX_r(:, ii_aoa)'*H_pseudo*A_TX_r));
                [~, ii_aoa] = max(abs((H_pseudo*A_TX_r(:, ii_aod))'*A_RX_r));
            end
            [~, ii_t] = max(abs(ifft((kron(A_TX_r(:, ii_aod).', A_RX_r(:, ii_aoa)')*Phi')*Pilots_res, Time_res_r, 2)));
        end
        %% Angular values and normallized time in [0, 1[
        aoa = Angle_RX_r(ii_aoa);
        aod = Angle_TX_r(ii_aod);
        t = (ii_t-1)/Time_res_r; % Time computed between 0 and 1
        %% Reconstruct pilots measurement
        h = reshape(A_RX_r(:, ii_aoa)*A_TX_r(:, ii_aod)', [], 1)*exp(-(0:N_fft-1)*(t*2i*pi));
        p = (Phi*reshape(A_RX_r(:, ii_aoa)*A_TX_r(:, ii_aod)', [], 1))*exp(-(0:N_fft-1)*(t*2i*pi));
        %% Dump data
        AoA = [AoA, aoa];
        AoD = [AoD, aod];
        ToF = [ToF, t];
        H   = [H, h(:)];
        P   = [P, p(:)];
        Alpha = P \ Pilots(:);
        %% Reconstruct channel
        Ch = reshape(H*Alpha, [N_RX, N_TX, N_fft]);
        CH{end+1} = Ch;
        TH(end+1) = (1-exp(-(amax/Noise_level)^2/2))^numel(Pilots_res_t_esp);
        %% Update pilot residual
        Pilots_res = reshape(Pilots(:) - P*Alpha, size(Pilots));
        Pilots_res_t = ifft(Pilots_res, Time_res_d, 2);
        Pilots_res_t_esp = (A_RTX_d*Phi')*Pilots_res_t;
        %% Compute noise level and the confidence threshold
        Noise_level = sqrt(1/(2*log(2)))*median(abs(Pilots_res_t_esp(:)), 1);
        Confidence_TH = Noise_level*sqrt(-2*log(1-Confidence_min^(1/numel(Pilots_res_t_esp))));
        %% Plot reconstruction and residual
        if bool_debug
            figure(2)
            plot(max(abs(ifft(reshape(P*Alpha, size(Pilots)), Time_res_d, 2)), [], 1), 'b', 'LineWidth', 1.5)
            figure(3)
            plot(max(abs(Pilots_res_t_esp), [], 1), 'b', 'LineWidth', 1.5), hold on
            plot([1, Time_res_d], repmat(Confidence_TH, 1, 2), 'r', 'LineWidth', 1); hold off
        end
        %% Recompute max
        [amax, imax] = max(abs(Pilots_res_t_esp(:)));
    end
    
end
    
function [Ch, AoA, AoD, ToF, Alpha] = ReconstructChannel(Pilots, Phi, N_RX, N_TX, D_w)
    % function R = ReconstructChannel(Pilots, Phi, N_RX, N_TX)
    %
    % INPUTS
    % Pilots - Measured antenna configurations with dimensions (N_Phi, N_fft)
    % Phi - Used antenna configurations with dimensions (N_Phi, N_RX*N_TX)
    % N_RX - Number of antenna elements in the receiver
    % N_TX - Number of antenna elements in the transmitter
    % Note: N_Phi is the number of used antenna configurations and N_fft the
    % number of subcarriers
    %
    % OUTPUTS
    % Ch - Reconstructed channel
    % AoA - Angles of arrival for the computed paths
    % AoD - Angles of departure for the computed paths
    % ToF - Times of flight for the computed paths
    % Alpha - Complex gains for the computed paths
    %% Parameters
    Time_sr_d = 4;             % Time sampling ratio
    Time_sr_r = 32;            % Time sampling ratio for refinement
    Angle_sr_d = 2;            % Angle sampling ratio for detection
    Angle_sr_r = 32;           % Angle sampling ratio for refinement
    Confidence = 0.98;         % Detection confidence
    PingPong_it = 3;           % Number of iteration to refine the angles
    bool_debug = false;        % Debug condition for plots
    %% Whitening
    if nargin > 4
        Pilots = D_w\Pilots;
        Phi = D_w\Phi;
    end
    %% Compute N_Phi and N_fft
    [N_Phi, N_fft] = size(Pilots);
    %% Derived parameters
    Time_res_d = Time_sr_d*N_fft;
    Time_res_r = Time_sr_r*N_fft;
    %% Compute angle responses
    Angle_RX_r = linspace(-pi, pi, N_RX*Angle_sr_r+1); Angle_RX_r(end) = [];
    Angle_TX_r = linspace(-pi, pi, N_TX*Angle_sr_r+1); Angle_TX_r(end) = [];
    A_RX_r = exp((0:N_RX-1).'*Angle_RX_r*1i)/sqrt(N_RX);
    A_TX_r = exp((0:N_TX-1).'*Angle_TX_r*1i)/sqrt(N_TX);
    %% Compute simple angle transform
    if Angle_sr_d < 2
        Angle_d = linspace(-pi, pi, N_RX*Angle_sr_d+1); Angle_d(end) = [];
        A_RX_d = SincBeam(N_RX, 1.1*2*pi/(N_RX*Angle_sr_d)).*exp((0:N_RX-1).'*Angle_d*1i);
        Angle_d = linspace(-pi, pi, N_TX*Angle_sr_d+1); Angle_d(end) = [];
        A_TX_d = SincBeam(N_TX, 1.1*2*pi/(N_TX*Angle_sr_d)).*exp((0:N_TX-1).'*Angle_d*1i);
    else
        Angle_d = linspace(-pi, pi, N_RX*Angle_sr_d+1); Angle_d(end) = [];
        A_RX_d = exp((0:N_RX-1).'*Angle_d*1i);
        Angle_d = linspace(-pi, pi, N_TX*Angle_sr_d+1); Angle_d(end) = [];
        A_TX_d = exp((0:N_TX-1).'*Angle_d*1i);
    end
    A_RTX_d = kron(A_TX_d.', A_RX_d');
    %% Dump pilots info into the residual pilots and convert it to time information
    Pilots_res = Pilots;
    Pilots_res_t = ifft(Pilots_res, Time_res_d, 2);
    Pilots_res_t_esp = (A_RTX_d*Phi')*Pilots_res_t;
    %% Compute noise level and the confidence threshold
    Noise_level = sqrt(1/(2*log(2)))*median(abs(Pilots_res_t_esp(:)), 1);
    Confidence_TH = Noise_level*sqrt(-2*log(1-Confidence^(1/numel(Pilots_res_t_esp))));
    %% Plot noise level and std margins
    if bool_debug
        figure(1)
        plot(max(abs(Pilots_res_t_esp)).', 'b', 'LineWidth', 1.5), hold on
        plot([1, Time_res_d], repmat(Confidence_TH, 1, 2), 'r', 'LineWidth', 1);hold off
    end
    %% Path substraction loop
    AoA = [];
    AoD = [];
    ToF = [];
    H   = [];
    P   = [];
    Alpha = [];
    [amax, imax] = max(abs(Pilots_res_t_esp(:)));
    while amax > Confidence_TH
        %% Imaging
        [ii_aoa, ii_aod, ii_t] = ind2sub([N_RX*Angle_sr_d, N_TX*Angle_sr_d, Time_res_d], imax);
        %% Extract time and spatial measurements
        t = (ii_t-1)/Time_res_d; % Time computed between 0 and 1
        Mt = Pilots_res_t(:, ii_t);
        %% Pseudochannel computation
        % These two options are similar but have different properties
        H_pseudo = reshape(Phi'*Mt, [N_RX, N_TX]);
    %     H_pseudo = reshape(Phi\Mt, [N_RX, N_TX]);
        %% First iteration estimation
        if N_RX > N_TX
            [~, ii_aoa] = max(abs((H_pseudo*A_TX_d(:, ii_aod))'*A_RX_r));
            [~, ii_aod] = max(abs((A_RX_r(:, ii_aoa)'*H_pseudo)*A_TX_r));
        else
            [~, ii_aod] = max(abs((A_RX_d(:, ii_aoa)'*H_pseudo)*A_TX_r));
            [~, ii_aoa] = max(abs((H_pseudo*A_TX_r(:, ii_aod))'*A_RX_r));
        end
        [~, ii_t] = max(abs(ifft((kron(A_TX_r(:, ii_aod)', A_RX_r(:, ii_aoa).')*Phi')*Pilots_res, Time_res_r, 2)));
        %% Ping pong
        for iter = 1:PingPong_it
            if N_RX > N_TX
                [~, ii_aoa] = max(abs((H_pseudo*A_TX_r(:, ii_aod))'*A_RX_r));
                [~, ii_aod] = max(abs(A_RX_r(:, ii_aoa)'*H_pseudo*A_TX_r));
            else
                [~, ii_aod] = max(abs(A_RX_r(:, ii_aoa)'*H_pseudo*A_TX_r));
                [~, ii_aoa] = max(abs((H_pseudo*A_TX_r(:, ii_aod))'*A_RX_r));
            end
            [~, ii_t] = max(abs(ifft((kron(A_TX_r(:, ii_aod).', A_RX_r(:, ii_aoa)')*Phi')*Pilots_res, Time_res_r, 2)));
        end
        %% Angular values and normallized time in [0, 1[
        aoa = Angle_RX_r(ii_aoa);
        aod = Angle_TX_r(ii_aod);
        t = (ii_t-1)/Time_res_r; % Time computed between 0 and 1
        %% Reconstruct pilots measurement
        h = reshape(A_RX_r(:, ii_aoa)*A_TX_r(:, ii_aod)', [], 1)*exp(-(0:N_fft-1)*(t*2i*pi));
        p = (Phi*reshape(A_RX_r(:, ii_aoa)*A_TX_r(:, ii_aod)', [], 1))*exp(-(0:N_fft-1)*(t*2i*pi));
        %% Dump data
        AoA = [AoA, aoa];
        AoD = [AoD, aod];
        ToF = [ToF, t];
        H   = [H, h(:)];
        P   = [P, p(:)];
        Alpha = P \ Pilots(:);
        %% Update pilot residual
        Pilots_res = reshape(Pilots(:) - P*Alpha, size(Pilots));
        Pilots_res_t = ifft(Pilots_res, Time_res_d, 2);
        Pilots_res_t_esp = (A_RTX_d*Phi')*Pilots_res_t;
        %% Compute noise level and the confidence threshold
        Noise_level = sqrt(1/(2*log(2)))*median(abs(Pilots_res_t_esp(:)), 1);
        Confidence_TH = Noise_level*sqrt(-2*log(1-Confidence^(1/numel(Pilots_res_t_esp))));
        %% Plot reconstruction and residual
        if bool_debug
            figure(2)
            plot(max(abs(ifft(reshape(P*Alpha, size(Pilots)), Time_res_d, 2)), [], 1), 'b', 'LineWidth', 1.5)
            figure(3)
            plot(max(abs(Pilots_res_t_esp), [], 1), 'b', 'LineWidth', 1.5), hold on
            plot([1, Time_res_d], repmat(Confidence_TH, 1, 2), 'r', 'LineWidth', 1); hold off
        end
        %% Recompute max
        [amax, imax] = max(abs(Pilots_res_t_esp(:)));
    end
    %% Reconstruct channel
    Ch = reshape(H*Alpha, [N_RX, N_TX, N_fft]);
        
end




function D_w = Whitening(Wtr,Ntrain,Lr)
    % obtain noise covariance matrix 
    Wtr = Wtr(:,1:Lr*Ntrain);
    blocks = reshape(Wtr, 64,4,Ntrain);
    blocks_l = num2cell(blocks,[1,2]);
    blocks_l = reshape(blocks_l,1,Ntrain);
    blocks_c = cellfun(@(x) x'*x, blocks_l, 'UniformOutput', false);
    C = blkdiag(blocks_c{:});
    D_w = chol(C);
end
        

function [H_freq, H_time,At,Ar] = gen_channel_ray_tracing(nc,Nr,Nt,Nfft,Ts,rolloff,Mfilter,chan_save_file)
    %function [H_freq, H_time] = gen_channel_ray_tracing(nc,Nr,Nt,Nfft,Ts,rolloff,Mfilter,chan_save_file)
    %
    % INPUTS
    % nc - index of channel from 1 to 10,000
    % Nr - number of receive antennas
    % Nt - number of transmit antennas
    % Nfft - size of the FFT
    % BW - bandwidth
    % chan_save_file (optional file name where channels are stored)
    %
    % OUTPUTS
    % H_freq is Nr x Nt x Nfft
    % H_time is Nr x Nt x L where L varies based on the delay spread
    %
    % Created June 9, 2020
    
    if nargin<8
        chan_save_file = 'target_channels.hdf5';
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% load the file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    channel_data = load(chan_save_file,'/channel_challenge');
    % channel_data has dimensions 8 x 100 x 10000 
    
    % Initialize parameters
    [N_ray_param, N_ray_max,N_chan_target] = size(channel_data);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Generate the channel based on the given parameters
    %% Filtering effects are considered
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    gain_db         = channel_data(1,:,nc);
    gain_lin        = 10.^(gain_db/10);
    gain_phase      = channel_data(8,:,nc)*pi/180;
    gain_comp       = gain_lin.*exp(1i*gain_phase);
    toa_raw         = channel_data(2,:,nc);
    toa             = toa_raw - min(toa_raw);
    %AoD_el          = channel_data(3,:,nc); % IGNORED
    AoD_az          = channel_data(4,:,nc);
    %AoA_el          = channel_data(5,:,nc); % IGNORED
    AoA_az          = channel_data(6,:,nc);
    is_LOS          = channel_data(7,:,nc);
    is_channel_LOS  = sum(is_LOS);
    
    % establish channel length, supposing we start a few samples before the
    % first ray
    early_samples = 3; % just to capture the beginning part of the sinc
    
    % compute rms delay spread
    mean_delay = toa*gain_lin' / norm(gain_lin);
    rms_delay_spread = sqrt( sum( (toa-mean_delay).^2 .* gain_lin) / norm(gain_lin));
    
    L = min(ceil(rms_delay_spread/Ts),ceil(Nfft/3)) + early_samples;
    
    % Generate At and Ar
    zt = (0:Nt-1)';
    zr = (0:Nr-1)';    
    a=gain_comp.*conj(gain_comp);
    [~,paths_id]=sort(a);
    np=5;
    
    At = zeros(Nt,N_ray_max);
    Ar = zeros(Nr,N_ray_max);
    for i=1:N_ray_max
        At(:,i) = exp(1i*pi*cos(AoD_az(i))*zt)/sqrt(Nt);
        Ar(:,i) = exp(1i*pi*cos(AoA_az(i))*zr)/sqrt(Nr);
    end
    
    % Generate time domain channel
    H_time = zeros(Nr,Nt,L);
    for ell=1:L        
        for paths = 1:N_ray_max
            H_time(:, :, ell) = H_time(:, :, ell) + gain_comp(paths) ... 
            * sinc((ell*Ts-toa(paths) + early_samples*Ts)/Mfilter/Ts)*cos(pi*rolloff*(ell*Ts-toa(paths) + early_samples*Ts)/Mfilter/Ts)/(1-(2*rolloff*(ell*Ts-toa(paths) + early_samples*Ts)/Mfilter/Ts)^2) ...
            * Ar(:,paths) * At(:,paths)';
        end %paths
    end %ell
    
    H_freq = zeros(Nr,Nt,Nfft);
    
    % Generate frequency domain channel
    for nt = 1:Nt
        for nr = 1:Nr
            H_freq(nr,nt,:) = fft(H_time(nr,nt,:),Nfft);
        end %nr
    end %nt    
    
    % Normalize the channel
    rho = Nt*Nr*Nfft/norm(H_freq(:),'fro')^2;
    H_freq = sqrt(rho)*H_freq;
    H_time = sqrt(rho)*H_time;
    
    
end

clc, clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initilization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%System parameters
Nt = 16; % Number of TX antennas
Nr = 64; % Number of RX antennas
Nbits= 4; % Number of bits available to represent a phase shift in the analog precoder/combiner.
Lt = 2;  % Number of TX RF chains 
Lr = 4;  % Number of RX RF chains 
Ns = 2;  % Number of data streams to be transmitted
Nfft=256 % Number of subcarriers in the MIMO-OFDM system
Pt=1 % Transmit power(mw)
Nfilter = 20;
Mfilter = 1; %no oversampling
rolloff = 0.8;
MHz = 1e6; 
fs = 1760*MHz; %Sampling frequency
Ts = 1/fs;

% Training parameters
Nc = 100; % Number of channels available in the channel data set up to 10000
Ntrain=100; % Number of training symbols to be received for each one of the available channels

data_set = 3; %2 or 3

switch data_set
    case 1
        SNR = -15;
    case 2
        SNR = -10;
    case 3
        SNR = -5;
    otherwise 
        error('data_set not found')
end %switch
snr = 10.^(SNR/10);

chan_save_file_hdf5 = strcat('RXpilot_SNR',num2str(data_set),'.hdf5');
chan_save_file_mat  = strcat('RXpilot_SNR',num2str(data_set));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate training precoders and combiners and matrix Phi in (13) in [1]
% They are generated as pseudorandom phase shifts. 
% We use a different training precoder and combiner for every training
% symbol to be transmitted through the same channel.
% We use the same set of training precoders and combiners for every channel.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nres = 2^Nbits; %resolution ofr the phase shifters
Phi=zeros(Ntrain*Lr,Nt*Nr);%Initialize measurement matrix Phi in [1,(10)] of size LrxNtNr.
                           % We have a different measurement matrix for every training symbol.
                           % Here we are initilizaing the measurement
                           % matrices for all the training symbols, which
                           % correspond to matrix Phi in (13)
randn(1);
tt=randi(Nres,[Nt Ntrain*Lt]);
for i = 1:Nres
   tt(tt==i) = exp(1i*2*pi*(i-1)/Nres);
end
Ftr=tt;  
        
tt=randi(Nres,[Nr Ntrain*Lr]);
for i = 1:Nres
   tt(tt==i) = exp(1i*2*pi*(i-1)/Nres);
end
Wtr=tt;    

Ftr = Ftr/sqrt(Nt);% pseudorandom training precoders (generating Ntrain, precoders for all the training symbols)
Wtr = Wtr/sqrt(Nr);% pseudoranmdom training combiners

save TrainingPrecoders.mat Ftr
save TrainingCombiners.mat Wtr
        
for i=1:Ntrain,
   signal = sqrt(1/2/Lt)*(sign(randn(Lt,1))+1i*sign(randn(Lt,1))); %training signal q (frequency flat)
   Phi((i-1)*Lr+(1:Lr),:)=kron(signal.'*Ftr(:,(i-1)*Lt+(1:Lt)).',Wtr(:,(i-1)*Lr+(1:Lr))');% Generate Phi in (13)
end



r = zeros(Ntrain*Lr,Nfft);%Initialize RX training symbols for one channel
R = zeros(Nc,Ntrain*Lr,Nfft);% Initialize RX training synbols for all the channels
Channels = zeros(Nc,Nr,Nt,Nfft);
nn = zeros(Lr*Ntrain,Nfft);% Initialize noise matrix at the RF combiner output for all the training symbols

tic
for j=1:Nc %Nc is number of channels 
    [Hk,H_time,At,Ar] = gen_channel_ray_tracing(j,Nr,Nt,Nfft,Ts,rolloff,Mfilter); 
    Channels(j,:,:,:) =  Hk;
    % Load channel parameters for channel j and build channel matrix including filtering
    % effects
   
    var_n = Pt/snr;
    Noise = sqrt(var_n/2)*(randn(Nr,Ntrain,Nfft)+1i*randn(Nr,Ntrain,Nfft));
    SNRaux = zeros(Nfft,1);

    for k = 1:Nfft % Generate RX pilots for every subcarrier
            for t=1:Ntrain
                Wrf_t = Wtr(:,(t-1)*Lr+(1:Lr));
                nn((1:Lr)+Lr*(t-1),k) = Wrf_t'*Noise(:,t,k);
            end
            signal_k = Phi*reshape(Hk(:,:,k),[],1);
            noise_k = nn(:,k);
            r(:,k) = Phi*reshape(Hk(:,:,k),[],1) + nn(:,k);
            SNRaux(k) = signal_k'*signal_k/(noise_k'*noise_k);
    end
    Average_SNR = 10*log10(mean(SNRaux))
    R(j,:,:)=r;
end
toc

save(chan_save_file_mat,'R');

%% training
%Calculate NMSE of the channel
TH_domain = linspace(0.7, 1, 2^12);
nmse = zeros(size(TH_domain));
iterations = zeros(Nc,1);
D_w = Whitening(Wtr,Ntrain,Lr);
size_R = size(R);
for ex=1:Nc
    Pilots = reshape(R(ex,:,:),size_R(2),size_R(3));
    H = reshape(Channels(ex,:,:,:), Nr,Nt,Nfft);
    [CH, TH] = ReconstructChannel_train(Pilots, Phi, Nr, Nt, D_w);
    TH = [TH, 0.7];
    for ii = 1:length(CH)
        th1 = TH(ii);
        th2 = TH(ii+1);
        Ch = CH{ii};
        nmse_Ch=NMSE_channel(Ch,H, Nfft);
        nmse(TH_domain >= th2 & TH_domain < th1) = nmse(TH_domain >= th2 & TH_domain < th1) + 10*log10(nmse_Ch);
    end
end
nmse = nmse/Nc;

figure(10);
plot(TH_domain, nmse,'r')
switch data_set
    case 1
        ylim([-19, -17.5])
    case 2
        ylim([-23, -20])
    case 3
        ylim([-25.5, -24])
end
xlabel("Detection threshold \theta")
ylabel("NMSE [dB]")

[amax, imax] = min(nmse);
fprintf("Best threshold for data_set=%i : %1.4f with %3.2fdB NMSE\n", data_set, TH_domain(imax), amax)

print('-f10', sprintf('Training_%i', data_set), '-dpng')



%% calculate NMSE
function nmse = NMSE_channel(H_hat,H, Nffc)
% INPUTS    
% H_hat - Channel estimate
% H - real channel
%OUTPUTS NMSE
    nmse = 0;
    den = 0;
    for subcarrier=1:Nffc
        sub = H_hat(:,:,subcarrier)-H(:,:,subcarrier);
        H_k = H(:,:,subcarrier);
        nmse = nmse + norm(sub(:),'fro')^2;
        den = den + norm(H_k(:),'fro')^2;
    end
    nmse = nmse/den;
end