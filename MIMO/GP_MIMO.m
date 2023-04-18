close all
clear all
clc

% MIMO Environment settings
M = 4;
K = 4;
SETTING_STRING = sprintf('M%d_K%d_alpha_0.2', M, K);
BANDWIDTH = 10e6;
TX_POWER_TOTAL = 1;
NOISE_POWER_dB = -75;
NOISE_POWER = 10^((NOISE_POWER_dB-30)/10) * BANDWIDTH;

CHANNELS_FILENAME = sprintf('Data_MIMO/effectiveChannelGains_test_%s.mat', SETTING_STRING);
OUTPUT_FILENAME = sprintf('Data_MIMO/GP_%s.mat', SETTING_STRING);

    
% Load the channel array
load(CHANNELS_FILENAME);
n_layouts = size(effectiveChannelGains, 1);
assert((size(effectiveChannelGains,2)==M) && (size(effectiveChannelGains,3)==K));

% Process direct-link and cross-link channels 
dl_all_layouts = [];
for i = 1:n_layouts
    dl_all_layouts = [dl_all_layouts; diag(squeeze(effectiveChannelGains(i,:,:))).'];
end
cl_all_layouts = effectiveChannelGains .* reshape(1-eye(K), 1, K, K);

power_controls_all = [];
range_prctiles = [];
% Solve one layout at a time
for i = 1:n_layouts
    disp(sprintf('MIMO %d of %d layouts', i, n_layouts));    
    % Construct the Geometric Programming instance
    % Define optimization variables
    gpvar t x(K);
    dl = reshape(dl_all_layouts(i,:), K, 1);
    cl = squeeze(cl_all_layouts(i,:,:));
    % Compute SINR terms
    signal_powers = dl .* x;
    interferences = cl * x + ones(K, 1)*(NOISE_POWER/TX_POWER_TOTAL);
    % Two types of constraints: 
    % 1. Inequality constraints for lowest SINR
    % 2. x sum up to 1 
    %    (use a posynomial inequality, can't have equality,
    %     but inequality should take equality at optimal solutions)
    power_limit = sum(x) <= 1;
    constr_array = [interferences * t <= signal_powers; power_limit];
    obj = t;
    [obj_value, solution, status] = gpsolve(obj, constr_array, 'max');
    % Verify and gather solutions
    assign(solution);
    sinrs = (dl .* x)./(cl*x + NOISE_POWER/TX_POWER_TOTAL);
    % Verify all power control are positive (Can't add into the constraint
    % due to error, but doesn't seem to need enforce)
    assert(min(x)>0)
    assert(sum(x)>0.95) % ensure enough power is allocated
    range_prctiles = [range_prctiles; (max(sinrs)-min(sinrs))/min(sinrs)];
    power_controls_all = [power_controls_all; reshape(x, 1, K)];
end

% Save the results
assert(size(power_controls_all, 1)==n_layouts && size(power_controls_all, 2)==K);
save(OUTPUT_FILENAME, 'power_controls_all');
range_prctiles
disp('GP completed!')