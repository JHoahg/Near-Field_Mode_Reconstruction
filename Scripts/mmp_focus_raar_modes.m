% mmp algorithm with additional coherent mode reconstruction
function [reconstruction, errors,  new_weight, adapted_weight, integrated_int, integrated_int_after] ...
    = mmp_focus_raar_modes(constraints, guess, iterations,...
    param, F_in)

if param.use_GPU == 1
    constraints = gpuArray(constraints);
    guess = gpuArray(guess);
end


%parse arguments
if nargin == 5
    fresnel_num = F_in;
else
    fresnel_num = param.F;
end

if(isfield(param,'do_errors') == 0)
    warning('errors are not logged.');
end

if(isfield(param,'focus_cut_off') == 0)
    warning('no focus cut off defined (p.focus_cut_off), i quit.');
    return
end

if(isfield(param,'is_synthetic_data') == 0)
    warning('type of input data(real or simulated) defined (p.is_synthetic_data) , i quit.');
    return
end

if(param.is_synthetic_data == 0)
    if(isfield(param,'supp') == 0)
        warning('no support defined, i quit.');
        return
    end
end


%prepare algorithm
h = waitbar(0, 'progress');
waitbar(0, h, ...
    sprintf('Preparing PropagatorGPUs...'));
num_planes = numel(fresnel_num);
% %PropagatorGPUs
for ii =1:num_planes
    props{ii} = feval(param.propagator, fresnel_num(ii), fresnel_num(ii), ...
        param.rec_width, param.rec_height,0,1);
    
end

for ii =1:num_planes
    inv_props{ii} = feval(param.propagator, -fresnel_num(ii), -fresnel_num(ii), ...
        param.rec_width, param.rec_height,0,1);
    
end

% %errors
if param.use_GPU == 1
    errors = gpuArray(zeros(iterations, num_planes, 2));
    integrated_int = gpuArray(zeros(iterations, num_planes, param.recon_modes));
    integrated_int_after = gpuArray(zeros(iterations, num_planes, param.recon_modes));
else
    errors = (zeros(iterations, num_planes, 2));
    integrated_int = (zeros(iterations, num_planes, param.recon_modes));
    integrated_int_after = (zeros(iterations, num_planes, param.recon_modes));
    
end

b_0 = param.b_0;
b_m = param.b_m;
b_s = param.b_s;

% variable for qr weights
new_weight = zeros(iterations, param.recon_modes);
adapted_weight = zeros(iterations, param.recon_modes);

if param.use_GPU == 1
    sum_of_modes = gpuArray(zeros(size(guess(:, :, 1))));
else
    sum_of_modes = (zeros(size(guess(:, :, 1))));
end

%start iterations
for ii = 1:iterations
    waitbar(ii / iterations, h, ...
        sprintf('iterations %d / %d ',ii, iterations));
    
    %     RAAR relaxation
    b = exp(-(ii/b_s)^3)*b_0 + (1 - exp(-(ii/b_s)^3))*b_m;
    
    guess_old = guess;
    %     temporarily abuse R_M for propagation results of individual distances
    if param.use_GPU == 1
        R_M = gpuArray(zeros(size(guess)));
    else
        R_M = (zeros(size(guess)));
    end
    %     ii
    %     order = randperm(num_planes);
    for jj = 1:num_planes
        
        % set to 0 for each distance
        sum_of_modes(:) = 0;
        
        for mm = 1:param.recon_modes
            guess(:, :, mm) = props{jj}.propTF(guess_old(:, :, mm));
            
            tmp = (abs(guess(:, :, mm)).^2);
            sum_of_modes =  sum_of_modes + tmp;
            % I_jm
            integrated_int(ii, jj, mm) = gather(sum(tmp(:).^2));
        end
        
        
        
        if(isfield(param,'do_errors') == 1)
            if param.do_errors == 1
                tmp = mid(constraints(:,:,jj), param) - ...
                    sqrt(mid(sum_of_modes,param));
                errors(ii, jj, 1) = sum(abs(tmp(:)).^2)./ (param.height*param.width);
            end
        end
        
        
        for mm = 1:param.recon_modes
            % per pixel mode weight
            guess(:, :, mm) = sqrt(abs(guess(:, :, mm)).^2 ./ sum_of_modes) ...
                .* (constraints(:,:,jj))...
                .* exp(1i .* angle(guess(:, :, mm)));
            
            tmp = (abs(guess(:, :, mm)).^2);
            integrated_int_after(ii, jj, mm) = gather(sum(tmp(:)));
            
            guess(:, :, mm) = inv_props{jj}.propTF(guess(:, :, mm));
            
            R_M(:,:,mm) = R_M(:,:,mm) + guess(:, :, mm);
        end %mode back prop
    end %planes
    R_M(:) = R_M(:) .* 1/num_planes; % calculate mean of all magnitude projections
    guess = R_M; %write back in guess variable
    P_M = guess;
    % reflect on M
    guess = 2*P_M - guess_old;
    R_M = guess;
    %
    %% P_S get unitary modes
    if(param.recon_modes >= 2)
        guess = (reshape(guess, param.rec_height * param.rec_width, param.recon_modes));
        
        modes = gather(guess);
        
        if(0) %switch for QR and SVD
            
            [bla2, R, V] = svd(bla2, 0);
            
            I_tot = param.I_tot;%gather(sum(integrated_int(ii, 1, :)))
            
            new_weight(ii, :) = gather(diag(R));
            new_weight(ii, :)./sum(new_weight(ii, :))
            
            adapted_weight(ii, :) = sqrt(new_weight(ii, :) ./ sum(new_weight(ii, :)) .* I_tot);
        else
            [modes, R] = qr(modes, 0);
            d_R = diag(R);
            [curr_weight, curr_ind] = sort(abs(d_R), 'descend');
            new_weight(ii, :) = gather(d_R(curr_ind));
            adapted_weight(ii, :) = new_weight(ii, :);
            %             bla2 = bla2(:, curr_ind);
        end
        
        
        if param.use_GPU == 1
            guess = gpuArray(reshape(modes, param.rec_height, param.rec_width, param.recon_modes));
        else
            guess = (reshape(modes, param.rec_height, param.rec_width, param.recon_modes));
        end
        
        %reweight modes
        for mm = 1:param.recon_modes
            if real(new_weight(ii, mm) > 0)
                guess(:, :, mm) = guess(:, :, mm) * adapted_weight(ii, mm);
            else
                guess(:, :, mm) = guess(:, :, mm) * abs(adapted_weight(ii, mm)) * exp(1i*pi);
            end
            %         fprintf('Intensity in mode %i : %f \n',mm, sum(sum(abs(guess(:,:,mm)).^2)) )
        end
    end
    % Reflect on S
    guess = 2*guess - R_M;
    
    % % new iterate
    for mm = 1:param.recon_modes
        guess(:, :, mm) = (b/2) * (guess(:, :, mm) + guess_old(:, :, mm)) + (1 -b)*P_M(:, :, mm);
    end
    
end % iterations

% get results back to host
reconstruction = gather(guess);
errors = gather(errors);
close(h);

end