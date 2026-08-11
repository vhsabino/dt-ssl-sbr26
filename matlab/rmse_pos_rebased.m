function r = rmse_pos_rebased(t, xs, ys, xt, yt, win_s, overlap)
%RMSE_POS_REBASED RMSE de posicao (m) em janelas deslizantes re-ancoradas
% (rebase de translacao a zero no inicio de cada janela, media ponderada
% por numero de amostras entre janelas). Esta e uma metrica de
% pos-processamento — a simulacao em si e free-run.
%
t0s = t(1):win_s*(1 - overlap):(t(end) - win_s);
if isempty(t0s), t0s = t(1); end   % split mais curto que a janela: 1 janela
e2 = []; w = [];
for t0 = t0s
    idx = find(t >= t0 & t <= t0 + win_s);
    if numel(idx) < 2, continue; end
    dx = (xs(idx) - xs(idx(1))) - (xt(idx) - xt(idx(1)));
    dy = (ys(idx) - ys(idx(1))) - (yt(idx) - yt(idx(1)));
    e2(end+1) = mean(dx.^2 + dy.^2);
    w(end+1)  = numel(idx);         
end
r = sqrt(sum(w .* e2) / sum(w));
end
