close all
clear all
clc

% D2D Environment settings
SETTING = 'B';
if SETTING == 'A'
    N_LINKS = 10;
    FIELD_LENGTH = 150;
    SHORTEST_DIRECTLINK = 5;
    LONGEST_DIRECTLINK = 15;
elseif SETTING == 'B'
    N_LINKS = 10;
    FIELD_LENGTH = 200;
    SHORTEST_DIRECTLINK = 20;
    LONGEST_DIRECTLINK = 30;
else
    N_LINKS = 15;
    FIELD_LENGTH = 300;
    SHORTEST_DIRECTLINK = 10;
    LONGEST_DIRECTLINK = 30;
end
SETTING_STRING = sprintf('N%d_L%d_%d-%dm', N_LINKS, FIELD_LENGTH, SHORTEST_DIRECTLINK, LONGEST_DIRECTLINK);
TX_POWER_dBm = 30;
NOISE_dBm_Hz = -169;
BANDWIDTH = 5e6;

TX_POWER = 10^((TX_POWER_dBm-30)/10);
NOISE_POWER = 10^((NOISE_dBm_Hz-30)/10) * BANDWIDTH;

CHANNELS_FILENAME = sprintf('Data_D2D/pl_test_%s.mat', SETTING_STRING);
OUTPUT_FILENAME = sprintf('Data_D2D/GP_%s.mat', SETTING_STRING);
    
% Load the channel array
load(CHANNELS_FILENAME);
n_layouts = size(effectiveChannelGains, 1);
assert((size(effectiveChannelGains,2)==N_LINKS) && (size(effectiveChannelGains,3)==N_LINKS));

% Process direct-link and cross-link channels 
dl_all_layouts = [];
for i = 1:n_layouts
    dl_all_layouts = [dl_all_layouts; diag(squeeze(effectiveChannelGains(i,:,:))).'];
end
cl_all_layouts = effectiveChannelGains .* reshape(1-eye(N_LINKS), 1, N_LINKS, N_LINKS);

power_controls_all = [];
range_prctiles = [];
% Solve one layout at a time
for i = 1:n_layouts
    disp(sprintf('D2D %d of %d layouts', i, n_layouts));    
    % Construct the Geometric Programming instance
    % Define optimization variables
    gpvar t x(N_LINKS);
    dl = reshape(dl_all_layouts(i,:), N_LINKS, 1);
    cl = squeeze(cl_all_layouts(i,:,:));
    % Compute SINR terms
    powers = dl .* x;
    interferences = cl * x + ones(N_LINKS, 1)*(NOISE_POWER/TX_POWER);
    % Two types of constraints: 
    % 1. Inequality constraints for lowest SINR
    % 2. Bound constraints on x
    ubound = x <= ones(N_LINKS, 1);
    constr_array = [interferences * t <= powers; ubound];
    obj = t;
    [obj_value, solution, status] = gpsolve(obj, constr_array, 'max');
    % Verify and gather solutions
    assign(solution);
    sinrs = (dl .* x)./(cl*x + NOISE_POWER/TX_POWER);
    % Verify all power control are positive (Can't add into the constraint
    % due to error, but doesn't seem to need enforce)
    assert(min(x)>0)
    range_prctiles = [range_prctiles; (max(sinrs)-min(sinrs))/min(sinrs)];
    power_controls_all = [power_controls_all; reshape(x, 1, N_LINKS)];
end

% Save the results
assert(size(power_controls_all, 1)==n_layouts && size(power_controls_all, 2)==N_LINKS);
save(OUTPUT_FILENAME, 'power_controls_all');
range_prctiles
disp('GP completed!')