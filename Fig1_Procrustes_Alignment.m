%% ── Stage 4 — Preprocessing and continuous PCA ──────────────────────
%  Build a template manifold from one reference session, then align every
%  session's mean manifold to it across a sweep of PC dimensionalities.
%  Stages 1-3 run per session and save '<mouse>_Manifold_<day>.mat'; this
%  stage loads those files, so it deliberately starts from a clean workspace.
close all; clear
mice = {'Animal_1';'Animal_2';'Animal_3';'Animal_4';'Animal_5';'Animal_6'; ...
        'Animal_7';'Animal_8';'Animal_9';'Animal_10';'Animal_11';'Animal_12'};
Session = {'Day_1';'Day_2';'Day_3';'Day_4';'Day_5';'Day_6';'Day_7';'Day_8'; ...
           'Day_9';'Day_10';'Day_11';'Day_12';

tem_mouse = mice{3};        % template mouse: animals 3 in fig. S1
tem_date  = Session{12};    % template session

% ---- Template session ----
load(strcat(tem_mouse,'_Manifold_',tem_date,'.mat'));
Template_manifold  = cat(3, PCA_data1{:});    % raw-trace PCs x time x trials
dTemplate_manifold = cat(3, dPCA_data1{:});   % deconvolved PCs x time x trials

clearvars -except Template_manifold dTemplate_manifold tem_date tem_mouse mice Session

% ---- Disparity sweep ----
% for door-open dataset: 10 s before → 15 s after
startindex = 1;
endindex   = 251;
winLen     = endindex - startindex + 1;
nSessions  = size(Session,1); % training session

dim = 50; % maximum alignment dimension

Disparity       = zeros(nSessions, dim);   % disparity(session, nPCs); column 1 unused (n starts at 2)
Align_manifold  = cell(1, dim);            % aligned session-mean manifolds, per nPCs
Align_trialmani = cell(dim, nSessions);    % aligned single-trial manifolds, per nPCs x session

for n = 2:dim
    align_manifold = zeros(winLen, n, nSessions);   % time x PCs x session (reset each n)

    % template mean over trials -> [time x n]
    dTemplate_manimean = transpose(mean(dTemplate_manifold(1:n, startindex:endindex, :), 3));

    for ss = 1:nSessions
        Date = Session{ss};
        %  Stages 1-3 run per session and save '<mouse>_Manifold_<day>.mat'
        load(strcat(mouse,'_Manifold_',Date,'.mat'), 'dPCA_data1');

        dPCA_data1      = cat(3, dPCA_data1{:});                             % PCs x time x trials
        dPCA_data1_mean = mean(dPCA_data1(1:n, startindex:endindex, :), 3);  % n x time


        % Align session mean to template, then reuse that transform on every trial
        [d, Z, transform] = procrustes(dTemplate_manimean, dPCA_data1_mean');
        Disparity(ss, n)        = d;
        align_manifold(:, :, ss) = Z;                                        % time x n

        T = transform.T; b = transform.b; c = transform.c;  % rotation/reflection, scale, translation
        nTrials = size(dPCA_data1, 3);
        data = cell(1, nTrials);
        for st = 1:nTrials
            trial    = squeeze(dPCA_data1(1:n, startindex:endindex, st));    % n x time
            data{st} = b * transpose(trial) * T + c;                         % time x n
        end
        data = cat(3, data{:});           % time x n x trials
        data = permute(data, [2 1 3]);    % n x time x trials
        Align_trialmani{n, ss} = data;
    end

    Align_manifold{n} = align_manifold;
end