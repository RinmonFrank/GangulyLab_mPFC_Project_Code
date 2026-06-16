%% ── Stage 1 — Preprocessing and continuous PCA ──────────────────────
%  Load one session, z-score the traces, run PCA on the full (continuous)
%  recording, pick how many PCs to keep, and orient each PC consistently.
load([SessionName,'.mat']); clear SessionName

% Normalize traces
ZsD = zscore(trace,  [], 2);          % raw calcium trace
ZsE = zscore(s_diff, [], 2);          % deconvolved event trace

sigma = 1; gsw = ceil(6*sigma + 1);
ZsE_s = zscore(smoothdata(s_diff, 2, 'gaussian', gsw), [], 2);   % smoothed deconvolved
BiE   = double(ZsE >= 1.5);           % binarized events (threshold = 1.5 SD)

% Continuous PCA
[coeff,  score,  ~, ~, explained]  = pca(ZsD');     % raw trace
[dcoeff, dscore, ~, ~, dexplained] = pca(ZsE_s');   % smoothed deconvolved

% PCs to keep = scree-plot elbow: point of maximum distance below the line
num_retained_pcs   = scree_knee(explained);
num_retained_pcs_d = scree_knee(dexplained);
fprintf('PCs to keep — raw: %d, deconvolved: %d\n', num_retained_pcs, num_retained_pcs_d);

% Orient PCs: each score should positively track mean population activity
reference   = mean(ZsD, 1)';
d_reference = mean(ZsE, 1)';
for d = 1:size(dcoeff, 2)
    if corr(score(:, d),  reference)   < 0, coeff(:, d)  = -coeff(:, d);  score(:, d)  = -score(:, d);  end
    if corr(dscore(:, d), d_reference) < 0, dcoeff(:, d) = -dcoeff(:, d); dscore(:, d) = -dscore(:, d); end
end
clear d sigma gsw reference d_reference

%% ── Stage 2 — Score sorting (PC phase + locking) ────────────────────
%  Phase of each timepoint in the PC1–PC2 plane, then test whether each
%  neuron's events lock to a preferred phase against a shuffled null.

phase  = atan2(score(:,2),  score(:,1));    % raw-trace PC phase
dphase = atan2(dscore(:,2), dscore(:,1));   % deconvolved PC phase

% locking check for single neuron
for i=1:size(BiE,1)
    p=phase(find(BiE(i,:)));
    var_p(i)=circ_var(p); mean_p(i)=circ_mean(p); std_p(i)=circ_std(p);
    MVL(i) = circ_r(p);
    prob_phase_firing(i,:)=histcounts(p,-pi:2*pi/40:pi,'Normalization','Probability');
    clear p H
end
clear i

% Sort neurons by preferred phase
[~, sorting_mean_angle] = sort(mean_p, 'descend');

phase_s  = smoothdata(phase,  'gaussian', 40);
dphase_s = smoothdata(dphase, 'gaussian', 100);

% Shuffle data
for i=1:size(BiE,1)
    for sh=1:1000
        p_sh=phase(randperm(length(phase),length(find(BiE(i,:)))));
        var_p_sh(i,sh)=circ_var(p_sh); std_p_sh(i,sh)=circ_std(p_sh);
        mean_p_sh(i,sh)=circ_mean(p_sh);
        MVL_sh(i,sh) = circ_r(p_sh);
        clear H p_sh
    end
end
clear i

% A neuron is "locked" if its MVL exceeds its own 99th-percentile null
locking=MVL; locking_sh=MVL_sh;
locking_sh_mean=mean(MVL_sh,2);
locking_sh_99=prctile(MVL_sh,99,2);
locking_sh_1=prctile(MVL_sh,1,2);
not_locked=find(locking<=locking_sh_99');
locked=setdiff(1:size(BiE,1),not_locked);
mean_p_locked=mean_p(locked);
std_p_locked=std_p(locked);

%% ── Stage 3 — Segmentation (door-open trials) ───────────────────────
%  Window each door-open event: 10 s before → 15 s after
%  (−100 to +150 frames @ 10 Hz = 251 frames per trial).

idx1 = find(behID == 1);                                                      % door-open events (behID 1)
Trial_idx1 = [eventstart(idx1)-100, eventstart(idx1), eventstart(idx1)+150];  % [pre(-10s), door(0), post(+15s)]
clear idx1

Event_data1 = {}; Event_s_data1 = {}; ZsD_data1 = {}; BiE_data1 = {};
PCA_data1 = {}; dPCA_data1 = {};

for ii = 1:size(Trial_idx1,1)
    idx1_pr = Trial_idx1(ii,1); idx1_d = Trial_idx1(ii,2); idx1_po = Trial_idx1(ii,3);
    if idx1_po > size(trace,2)
        disp('Skip last trial')
        break
    end
    ZsD_data1(ii)     = {ZsD(:,idx1_pr:idx1_po)};               % raw trace, 10s pre -> 15s post door-open
    Event_data1(ii)   = {ZsE(:,idx1_pr:idx1_po)};               % deconvolved event
    Event_s_data1(ii) = {ZsE_s(:,idx1_pr:idx1_po)};             % smoothed deconvolved event
    BiE_data1(ii)     = {BiE(:,idx1_pr:idx1_po)};               % binarized event
    PCA_data1(ii)     = {transpose(score(idx1_pr:idx1_po,:))};  % raw-trace PCs x time
    dPCA_data1(ii)    = {transpose(dscore(idx1_pr:idx1_po,:))}; % deconvolved PCs x time
end
clear ii idx1_pr idx1_d idx1_po

Event_data1 = cat(3,Event_data1{:}); Event_s_data1 = cat(3,Event_s_data1{:});
ZsD_data1 = cat(3,ZsD_data1{:}); BiE_data1= cat(3,BiE_data1{:});
Event_data1_mean = mean(Event_data1,3);
