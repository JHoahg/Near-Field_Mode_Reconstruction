function [f_constraints] =...
    multiple_beam_distances(object, probe, F, p, weights, noise)
%% build measurements
% assemble constraints we need:
if(nargin <= 4)
weights = ones([1 (p.num_modes)]);
noise = 0;
end

if(numel(object) ~= numel(probe{1}))
    warning('Sizes of probe and object do not match.');
    return
end 


f_constraints = gpuArray(zeros(p.rec_height, p.rec_width, numel(F)));
% propagators = Propagator.empty(numel(F),0);
for ii = 1:numel(F)
    prop = PropagatorGPU(F(ii), F(ii), size(object,2), size(object,1),0,1);
    for jj = 1:p.main_modes
        %         tmp =  exp(1i .* zeros(p.width2, p.height2));
        %
        %         tmp((p.width2 - p.width)/2 + 1 : (p.width2 - p.width) / 2 + p.width, ...
        %             (p.height2 - p.height)/2 + 1 : (p.height2 - p.height) / 2 + p.height) ...
        %             = probe{jj} .* object;
%         tmp = pad_smoothly(probe{jj} .* object, p.width2/p.width);%JH
%         tmp = pad_smoothly(probe{jj} .* object, p.width2/p.width); %ALR
        %         tmp = pad_to_size(probe{jj} .* object, p.width2, p.height2);

        
         if p.norm == 1
           tmp = probe{jj};
           ampl = sqrt(sum(sum(abs(tmp).^2)));
           tmp = tmp ./ampl .* object .* sqrt(abs(weights(jj)));
         else
            tmp = probe{jj} .* object;
         end
            
        tmp = prop.propTF(tmp);
        
        tmp = mid(tmp, p.rec_height, p.rec_width);
        
        disp(sprintf('Intensity in mode %i : %f',jj, sum(abs(tmp(:)).^2) ))
        
        if(0)
        figure(999)
        imagesc(abs(tmp))
        drawnow;
        export_fig(sprintf('./figs/propagated_mode_%d_F_%g.pdf', jj, F(ii)));
        end
        
        tmp = (abs(tmp)).^2;
        
        tmp = mid(tmp, p.rec_height, p.rec_width);
        
        % fringes outside the detector area get cut off
        %         f_constraints((p.width2 - p.width)/2 + 1 : (p.width2 - p.width) / 2 + p.width, ...
        %             (p.height2 - p.height)/2 + 1 : (p.height2 - p.height) / 2 + p.height, ii) = ...
        %             tmp2((p.width2 - p.width)/2 + 1 : (p.width2 - p.width) / 2 + p.width, ...
        %             (p.height2 - p.height)/2 + 1 : (p.height2 - p.height) / 2 + p.height);
        %
        
        %use all fringes
        
%         sum(tmp(:))
        f_constraints(:, :, ii) =  f_constraints(:, :, ii) + tmp;
        
        %         figure(2 *(ii*p.num_modes + jj) -1)
        %         imagesc(f_constraints(:,:,ii))
        %         fname = ['F_', num2str(F(ii)), '_mode_', num2str(jj)];
        %         title(fname); colorbar;
        % %         print('-depsc', [fname, '.pdf'])
        %
        %         figure(2*(ii*p.num_modes + jj))
        %         imagesc(tmp2)
        %         fname = ['contribution_of_mode_', num2str(jj), '_F_', num2str(F(ii))];
        %         title(fname); colorbar;
        % %         print('-depsc', [fname, '.pdf'])
        
    end %modes
    
    if noise == 1
%         f_constraints(:, :, ii) = f_constraints(:, :, ii) ./ sum(sum(f_constraints(:, :, ii)));
%         f_constraints(:, :, ii) = imnoise((f_constraints(:, :, ii)*p.num_photons.* numel(f_constraints(:, :, ii)))*1e-12, 'poisson')*1e12;
%         f_constraints(:, :, ii) = f_constraints(:, :, ii) ./ sum(f_constraints(:, :, ii)) .* numel(f_constraints(:, :, ii));
%         disp('bla')
          f_constraints(:, :, ii) = imnoise((f_constraints(:, :, ii))*1e-12, 'poisson')*1e12;
      
    end    
            
            %             gamma = 1e9;
%             tmp = 1e12 .* imnoise(tmp .* gamma.* 1e-12, 'poisson') .* 1/gamma;

if(0)
    figure
        imagesc(abs(mid(f_constraints(:, :, ii),p))); title(sprintf('constraint at F=%f', F(ii)))
        
        disp(sprintf('Intensity in measuremant %i : %f',ii, sum(sum(abs(f_constraints(:, :, ii)))) ))
        drawnow;
end
if(0)
        export_fig(sprintf('./figs/incoherent_measurement_at_F_%g.pdf', F(ii)));
    end
end %distances

f_constraints = gather(f_constraints);
end
