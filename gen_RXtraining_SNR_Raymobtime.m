% Script to generate M received training symbols for every MIMO channel in the matrix obtained from NYUSIM channel simulator
% - A fully connected hybrid MIMO architecture is assumed at the TX and RX. 
% - The received training pilot is obtained at the output of the ADCs for the different RF chains. 
% - Only analog precoding/combining during training is used. 
% - ULAs are assumed
% References:
% [1] J. Rodríguez-Fernández, N. González-Prelcic, K. Venugopal and R. W. Heath, "Frequency-Domain Compressive Channel Estimation for Frequency-Selective Hybrid Millimeter Wave MIMO Systems," IEEE Transactions on Wireless Communications, vol. 17, no. 5, pp. 2946-2960, May 2018.

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
Nc = 1000; % Number of channels available in the channel data set up to 10000
Ntrain=100; % Number of training symbols to be received for each one of the available channels

data_set = 1; %2 or 3

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
rng(1);
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

nn = zeros(Lr*Ntrain,Nfft);% Initialize noise matrix at the RF combiner output for all the training symbols

tic
for j=1:Nc %Nc is number of channels 
    Hk = gen_channel_ray_tracing(j,Nr,Nt,Nfft,Ts,rolloff,Mfilter); 
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

save(chan_save_file_mat,'R','-v7.3');

%%
load RXpilot_SNR1;

% Write the HDF5 file. Note that we have to write the real and imaginary
 %parts separately due to the file format
 if ~isfile(chan_save_file_hdf5)
     h5create(chan_save_file_hdf5, '/training_data_real', size(R));
     h5create(chan_save_file_hdf5, '/training_data_imag', size(R));
 end
 realR = real(R);
 imagR = imag(R);
 h5write(chan_save_file_hdf5, '/training_data_real', realR);
 h5write(chan_save_file_hdf5, '/training_data_imag', imagR);

filename = chan_save_file_hdf5;
fileID = H5F.create(filename,'H5F_ACC_TRUNC','H5P_DEFAULT','H5P_DEFAULT');
datatypeID = H5T.copy('H5T_NATIVE_DOUBLE');
dims = size(R);
plistID = H5P.create('H5P_DATASET_CREATE'); % create property list
dataspaceID = H5S.create_simple(3,fliplr(dims),[]);
chunk_size = min([256 256 256], dims); % define chunk size 
H5P.set_chunk(plistID, fliplr(chunk_size)); % set chunk size in property list
dsetname = 'training_data_real';  
datasetID = H5D.create(fileID,dsetname,datatypeID,dataspaceID,plistID);
H5D.write(datasetID,'H5ML_DEFAULT','H5S_ALL','H5S_ALL','H5P_DEFAULT',realR);
dsetname = 'training_data_imag';  
datasetID = H5D.create(fileID,dsetname,datatypeID,dataspaceID,plistID);
H5D.write(datasetID,'H5ML_DEFAULT','H5S_ALL','H5S_ALL','H5P_DEFAULT',imagR);

H5D.close(datasetID);
H5S.close(dataspaceID);
H5T.close(datatypeID);
H5F.close(fileID);

%% example of reading the file to test correctness
rR = h5read(chan_save_file_hdf5,'/training_data_real');
iR = h5read(chan_save_file_hdf5,'/training_data_imag');
Rhat = rR+1i*iR;
norm(abs(R(:) - Rhat(:)),'fro')


