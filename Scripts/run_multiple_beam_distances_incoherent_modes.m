%% incoherent modes reconstruction script
% change folder to Scripts before executing, sadly one can not find out
% in Matlab where a plain script lies in the filesysytem :(
% http://de.mathworks.com/matlabcentral/newsreader/view_thread/269433
working_dir = '.';
cd(working_dir)
%data dir
TB_path = '../Tools';
addpath(genpath(TB_path));

path = getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH'); %clears path
path = [path ':/usr/local/cuda/lib'] % new path
setenv('LD_LIBRARY_PATH', path);
clear path
% 
set(0,'DefaultFigureColor','w')
cmap = flipud(colormap('bone(512)'));
set(0,'DefaultFigureColormap', cmap);
close 1

cmd = sprintf(['axis image; colorbar;  set(gcf,''PaperPosition'',[0 0 5 5], ''PaperSize'', [5 5]);',...
    ' set(gca,''xtick'',[],''ytick'',[])']);
set(0,'DefaultImageCreateFcn', cmd)
set(groot,'defaultLineLineWidth',1.5)
clear cmap cmd;
%% Simulation parameters
p.width = 1200;
p.height = 1200;
p.width2 = 2048;
p.height2 = 2048;
p.rec_width = 2048;
p.rec_height = 2048;

p.lambda = 1.24/8; %nm E = 8 keV
p.pixel_size_det = 700; %nm
p.pixel_size = 50;
% p.det_distances = [0.36 0.35 0.34 0.33]*1e9;
% nur 8 ebenen
% p.det_distances = [0.325 0.328 0.331 0.335 0.338 0.341 0.345 0.35]*1e9;

% % nur 6 ! ebenen
% p.det_distances = [0.325 0.331 0.335 0.341 0.345 0.35]*1e9;

% nur 5 !!! ebenen
% p.det_distances = [0.325 0.335 0.341 0.345 0.35]*1e9;


p.det_distances = [0.325 0.328 0.331 0.335 0.338 0.341 0.345 0.35 0.36 0.37]*1e9;
% p.det_distances = [0.325 ]*1e9;
% p.det_distances = [0.33]*1e9;

p.z1 = 0.32e9;
p.distances = p.det_distances - p.z1; 

% define fresnel number wrt to 1st det plane
p.F = p.pixel_size^2./(p.lambda .* p.distances);

%beam parameter
p.w0 = 700; %31.8; %nm
p.focus_cut_off = 200*p.w0;
p.is_synthetic_data = 1;
p.use_noise = 0;
p.num_photons = 1; %photons per pixel

if(1)%p.rec_height < 4097
    p.propagator = @PropagatorGPU;
    p.use_GPU = 1;
else 
    p.propagator = @Propagator;
    p.use_GPU = 0;
end
p.b_0 = 0.99;
p.b_m = 0.5;
p.b_s = 1150;

p.unitary = 1;
p.main_modes = 3;
p.recon_modes = 3;
p.shifts_per_mode = 0;
p.num_modes = p.main_modes*(1 + p.shifts_per_mode);

% all photons have to be distributed among the modes
all_photons = p.width2 * p.height2 * p.num_photons; 
% p.mode_weight = all_photons*[0.5 0.3 0.2];
p.mode_weight = 4e6*[0.5 0.3 0.2];
p.I_tot = sum(p.mode_weight);


% p.init_weight = [0.55 0.31 0.22];% 0.01];
p.init_weight = [1/3 1/3 1/3];% 0.01];
p.init_weight = p.init_weight ./ sum(p.init_weight) * all_photons;
% p.init_weight = 4e6*[0.9 0.1];
warning('summed mode weight is %f', sum(p.mode_weight))
p



%% get images, build multimodal probe
lower_pha = -0.4;
upper_pha = 0.4;
lower_amp = 0.8;
upper_amp = 1.2;
gauss_fwhm = 0;
clear probe
probe{1} = prepare_probe(['./house.png'], ['./lakehouse.png'], lower_pha, upper_pha, lower_amp, upper_amp, p, gauss_fwhm);
probe{2} = prepare_probe(['./mandrill.png'], ['./durer.png'],  lower_pha, upper_pha, lower_amp, upper_amp, p, gauss_fwhm);
probe{3} = prepare_probe(['./parrots.png'], ['./lighthouse.png'], lower_pha, upper_pha, lower_amp, upper_amp, p, gauss_fwhm);
probe{4} = prepare_probe(['./frog.tif'], ['./kathrine.tif'], lower_pha, upper_pha, lower_amp, upper_amp, p, gauss_fwhm);
%% 
if(p.unitary) % QR toggle
    %% QR-Zerlegung von den moden
    % prepare input
    A_in = (zeros(p.height2 * p.width2, size(probe, 2)));
    
    for ii=1:size(probe, 2) 
        A_in(:, ii) = reshape((probe{ii}), p.height2 * p.width2, 1);
    end
    %% get unitary modes
    tic;
    [ortho_probe, R] = qr((A_in), 0);
    toc;
    %%
    clear probe
    for ii=1:p.num_modes
        % leave out first mode, looks ugly
        if(real(R(ii+1, ii+1)) < 0)
            probe{ii} = reshape((ortho_probe(:, ii +1 )), p.height2, p.width2) .* exp(1i*pi);
        else
            probe{ii} = reshape((ortho_probe(:, ii +1 )), p.height2, p.width2);
        end
    end
end%for QR

%%
% coherent_probe = zeros([p.height2 p.width2]);
if (1)
for ii = 1:p.num_modes
    figure(2*ii -1 + 0)
    imagesc(mid(angle(probe{ii}),p))
    fname = (['phases of mode ', num2str(ii)])
    
    title(fname); colorbar;
%     export_fig(sprintf('amp_ortho_mode_%i.pdf', ii), '-CMYK', '-q101')
    
    
    figure(2*ii + 0)
    imagesc(mid(abs(probe{ii}),p))
    fname = (['amplitudes of mode ', num2str(ii)])
    title(fname); colorbar;
%     export_fig(sprintf('pha_ortho_mode_%i.pdf', ii), '-CMYK', '-q101')
end
end

%% build measurements
warning('maximum pixelsize from criterion')
pxs = p.distances .* p.lambda / (p.pixel_size * p.width2)
tic
f_constraints = num2cell(1);
p.norm = 1;

f_constraints{1} = ...
    multiple_beam_distances(ones(p.width2), probe, p.F, p, p.mode_weight, p.use_noise);
toc
%%
p.init_weight = rand([1 p.recon_modes]);
p.init_weight =  sort(p.init_weight ./sum(p.init_weight) * all_photons, 'descend');
p.init_weight ./all_photons
%% create guess
p.b_0 = 0.5;
p.b_m = 0.99;
p.b_s = 50;

p.do_errors = 1;
guess = rand(p.height2 * p.width2, p.recon_modes) .* exp(1i*rand(p.height2 * p.width2, p.recon_modes));
% get unitary modes
[guess, R] = qr((guess), 0);
    
guess = reshape(guess, p.height2, p.width2, p.recon_modes);
    
for ii = 1:p.recon_modes
    ampl = sqrt(sum(sum(abs(guess(:,:,ii)).^2)));
    guess(:,:,ii) = guess(:,:,ii) ./ ampl .* sqrt(p.init_weight(ii)); 
    disp(sprintf('Intensity in mode %i : %f',ii, sum(sum(abs(guess(:,:,ii)).^2)) ))
end


tic
iterations{1} = 100;
[reconstruction{1}, errors{1}, new_weight{1}, adapted_weight{1},  integrated_int{1},  integrated_int_after{1}] = ...
    mmp_focus_raar_modes(sqrt(f_constraints{1}(:,:,:)),...%beware of intensity...
                  guess,...
                   iterations{1}, p, p.F);
rec_number = 1;               

toc

%%
p.b_0 = 0.99;
p.b_m = 0.99;
p.b_s = 500;
for ii = 2:4
    iterations{ii} = 5000;
    tic
    sprintf('start %s', char(datetime, 'yyyy_MM_dd_''T''HH_mm_ss'))
    [reconstruction{ii}, errors{ii}, new_weight{ii}, adapted_weight{ii}] = ...
        mmp_focus_raar_modes(sqrt(f_constraints{1}(:,:,:)),...%beware of intensity...
        reconstruction{ii-1},...
        iterations{ii}, p, p.F);
%     reco_prelim = reconstruction{ii};
%     fname = sprintf('%s/reconstruction_04_04_%f_iter_%s.mat',...
%             p.info, sum(cellfun(@(x) sum(x(:)),iterations)),...
%             char(datetime, ('yyyy_MM_dd_''T''HH_mm_ss')));
%     save(fname, 'reco_prelim', 'p', 'errors')
    toc
    char(datetime, 'yyyy_MM_dd_''T''HH_mm_ss')
    rec_number = ii;
end
