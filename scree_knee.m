function n = scree_knee(explained)
% PCs to keep = scree-plot elbow: point of maximum distance below the line
% joining the first and last explained-variance values.
x = explained(:)'; k = numel(x);
line = x(1) + (x(end) - x(1)) / (k - 1) * (0:k-1);
[~, n] = min(x - line);
end