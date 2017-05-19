%% plot modes
for ii = 1 : p.recon_modes
    figure
    tmp = mid(reconstruction{rec_number}(:,:,ii),p) *-1;
    imagesc(abs(tmp))
    title(sprintf('amplitude mode %i', ii))
    drawnow
    disp(sprintf('Intensity in mode %i : %e',ii, sum(sum(abs(reconstruction{rec_number}(:,:,ii)).^2)) ))
    figure
    imagesc(angle(tmp))
    title(sprintf('phases mode %i', ii))
    drawnow
end

%%
weights = new_weight{1};
all_errors = errors{1};
% a_weights = adapted_weight{1};
for ii = 2:rec_number
    weights = [weights; new_weight{ii}];
    all_errors = [all_errors; errors{ii}];
    %     a_weights = [a_weights; adapted_weight{ii}];
end
%%
figure
hold on
for ii = 1:p.recon_modes
    plot(weights(:,ii).^2, 'DisplayName', sprintf('M. %i', ii), 'LineWidth', 2)
    
    if(p.is_synthetic_data == 1 & ii < p.num_modes)
        plot(1:size(weights,1), ones(size(weights,1),1) .* (p.mode_weight(ii)),...
            'DisplayName', sprintf('target  %i', ii), 'LineWidth', 2)
    end
    %     plot((a_weights(:,ii).^2),'DisplayName', sprintf('adapted weight'))
end
% plot(1:size(weights, 1), ones(size(weights, 1),1))
if(p.is_synthetic_data == 1)
    plot(1:size(weights,1), ones(size(weights,1),1) .* sum(p.mode_weight(p.recon_modes:end)),...
        'DisplayName', sprintf('rest'), 'LineWidth', 2)
end
hold off
%  set(gca,'xscale','log')
 xticks([1,10,100,1000,1e4])
legend('show', 'Location', 'best')
title('QR weight')
%% weights relative to init
figure
hold on
for ii = 1:p.recon_modes
    plot(weights(:,ii).^2 ./ p.mode_weight(ii), 'DisplayName', sprintf('\\Psi %i', ii), 'LineWidth', 2)
end

plot(1:size(weights, 1), ones(size(weights, 1),1), 'DisplayName','', 'LineWidth', 2)
% if(p.is_synthetic_data == 1)
%  plot(1:size(weights,1), ones(size(weights,1),1) .* sum(p.mode_weight(p.recon_modes:end)),...
%           'DisplayName', sprintf('rest'), 'LineWidth', 2)
% end
hold off
%  set(gca,'xscale','log')
%  xticks([1,10,100,1000,1e4])
 ylim([0 3])
legend('show', 'Location', 'best')
%  ylim([0 10])
title(' relative QR weight')
%% errors
figure
hold on
for ii = 1:numel(p.F)
    plot(all_errors(:,ii,1), 'DisplayName', sprintf('M. %i', ii), 'LineWidth', 2)
end

 set(gca,'yscale','log')
  set(gca,'xscale','log')
% xlim([5 numel(all_errors(:,ii,1))])
hold off
legend('show', 'Location', 'best')
% ylim([1e-1 1])


%% test orthogonality
%% QR-Zerlegung von den moden
% prepare input
A_in = (zeros(p.rec_height * p.rec_width, p.recon_modes));

for ii=1:p.recon_modes
    rec_probe{ii} = mid(reconstruction{rec_number}, p.rec_height, p.rec_width);
    A_in(:, ii) = reshape((rec_probe{ii}), p.rec_height * p.rec_width, 1);
end
%% get R matrix of reconstruction
tic;
[ortho_probe, R] = qr((A_in), 0);
toc;
log10(abs(R))
%% more ortho test
%%
F = inf;
for ii = 1:1
    prop = PropagatorGPU(F, F, p.rec_width, p.rec_height,1);
    for jj = 1:p.recon_modes
        
        tmp2 = rec_probe{jj};
        
        tmp2 = prop.propTF(tmp2);
        
        single_mode{jj} = tmp2;
        
        %         figure(2 *(ii*p.num_modes + jj) -1)
        %         imagesc(f_constraints(:,:,ii))
        %         fname = ['F_', num2str(F(ii)), '_mode_', num2str(jj)];
        %         title(fname); colorbar;
        %         print('-depsc', [fname, '.pdf'])
        %
        figure(2*(ii*p.num_modes + jj))
        imagesc(abs(tmp2))
        fname = ['contribution_of_mode_', num2str(jj), '_F_', num2str(F)];
        title(strrep(fname, '_', ' ')); colorbar;
        %         print('-depsc', [fname, '.pdf'])
        
    end %modes
end %distances
%% calc scalar prod
corr_fac = zeros(p.num_modes);
for ii = 1:p.num_modes
    for jj = ii : p.recon_modes
        res = gather(sum(sum(rec_probe{ii} .* conj(rec_probe{jj}))));
        %        res = sum(sum(abs(single_mode{ii}) .* abs(single_mode{jj})));% / numel(single_mode{ii});
        %        res = sum(sum(abs(single_mode{ii}) .* abs(single_mode{jj})));
        %        res = corr2(abs(single_mode{ii}), abs(single_mode{jj}));
        corr_fac(ii, jj) = res;
        corr_fac(jj, ii) = res;
    end
end

log10(abs(corr_fac))
%% find phase offset
    mm = 3;
    tmp_probe = mid((probe{mm}),p);
    figure;
    imagesc(angle(tmp_probe))
    title( sprintf('original mode %i',mm ));
    colormap hsv

for jj = rec_number
    
    tmp_rec = mid((reconstruction{jj}(:,:,mm)), p);
   
    
    ii = 1;
    range = -pi:pi/100:pi;
    for of = range
        bla = image_diff(of, tmp_rec, tmp_probe);
        result(ii) = sum(abs(bla(:)).^2)/(1200*1200) ;
        ii = ii+1;
    end
    
    
    offset_guess = range(find(result == min(result)));
    %%
    
    f_handle = @(x)image_diff(x,tmp_rec, tmp_probe);
    
    [offset, chi] = lsqnonlin(f_handle, offset_guess);
    
%     figure;
%     imagesc(angle(tmp_rec * exp(1i*offset)));
%         colormap hsv
    figure;
    imagesc(abs(tmp_rec));
    
    delta = sum(sum(abs(angle(tmp_rec * exp(1i*offset)) - angle(tmp_probe))))...
        /(p.height*p.width);
    title( sprintf('Probe \\Delta=%f rec number %i', (delta), jj));
    set(gcf,'color','white');
  
    if(1)
        figure
        imagesc(angle(tmp_rec * exp(1i*offset)) - angle(tmp_probe))
    end
    drawnow
end




