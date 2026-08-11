function E = check_excitation_trans(opts)
%CHECK_EXCITATION_TRANS Diagnostico de excitacao TRANSLACIONAL por evento,
% separado para vx e vy (analogo a check_excitation de omega).
%
% Comando de corpo: move_x (=vx) e move_y (=vy) de commands.parquet.
% Velocidade MEDIDA de corpo: derivada (sgolay) da posicao da VISAO
% (position_x/position_y em mm -> m), rotacionada do mundo p/ o corpo por
% theta=unwrap(position_w):
%   vx_b =  cos(th)*vx_w + sin(th)*vy_w ;  vy_b = -sin(th)*vx_w + cos(th)*vy_w
% (mesma rotacao de build_idData_per_dof, porem com velocidade derivada da
% posicao em vez da velocity_* logada — instrucao da Fase 1).
%
% Por evento e por DOF: std do comando, n. de frequencias significativas do
% comando, atraso Td0 da maxima xcorr cmd->vel medida e o valor dessa xcorr,
% e segundos ativos (|cmd|>VThresh). USO: E = check_excitation_trans();

arguments
    opts.DataRoot (1,:) char    = fullfile('data','extracted')
    opts.Label    (1,:) char    = ''      % '' = todos
    opts.Team     (1,:) char    = 'allies'
    opts.RobotId  (1,1) double  = NaN
    opts.Ts       (1,1) double  = 1/60
    opts.SgWin    (1,1) double  = 7
    opts.SgOrder  (1,1) double  = 2
    opts.MaxLagS  (1,1) double  = 1.0
    opts.VThresh  (1,1) double  = 0.3      % m/s, "ativo"
    opts.SaveCsv  (1,1) logical = false
end

repo_root = bootstrap_paths();
data_root = opts.DataRoot;
if ~startsWith(data_root, repo_root), data_root = fullfile(repo_root, data_root); end
S = list_splits(data_root, 'Label', opts.Label);
S = S(strcmp(S.chunk, 'commands'), :);
if isempty(S), error('Nenhum chunk commands em %s (Label=%s).', data_root, opts.Label); end

rows = {};
for i = 1:height(S)
    C = parquetread(S.path{i});
    rid = opts.RobotId; if isnan(rid), rid = mode(double(C.robot_id)); end
    C = C(double(C.robot_id)==rid, :);
    rp = fullfile(fileparts(S.path{i}), 'processed_robots.parquet');
    if height(C) < 30 || ~isfile(rp), continue; end
    R = parquetread(rp);
    R = R(strcmp(R.team, opts.Team) & double(R.robot_id)==rid, :);
    if height(R) < 30, continue; end

    t  = C.timestamp_event; dur = t(end)-t(1); Tsm = median(diff(t));
    t0 = max(t(1), R.timestamp_event(1)); t1 = min(t(end), R.timestamp_event(end));
    tg = (t0:opts.Ts:t1)'; if numel(tg) < 30, continue; end
    ux = interp1(t, C.move_x, tg, 'previous', 0);
    uy = interp1(t, C.move_y, tg, 'previous', 0);
    px = interp1(R.timestamp_event, R.position_x/1000, tg, 'linear');
    py = interp1(R.timestamp_event, R.position_y/1000, tg, 'linear');
    th = interp1(R.timestamp_event, unwrap(R.position_w), tg, 'linear');
    win = min(opts.SgWin, 2*floor((numel(tg)-1)/2)+1);
    if win > opts.SgOrder
        px = sgolayfilt(px, opts.SgOrder, win); py = sgolayfilt(py, opts.SgOrder, win);
    end
    vxw = gradient(px)/opts.Ts; vyw = gradient(py)/opts.Ts;
    vxb =  cos(th).*vxw + sin(th).*vyw;
    vyb = -sin(th).*vxw + cos(th).*vyw;

    [Tx, Cx] = xcorr_delay(ux, vxb, opts.Ts, opts.MaxLagS);
    [Ty, Cy] = xcorr_delay(uy, vyb, opts.Ts, opts.MaxLagS);
    rows{end+1} = {S.log{i}, S.label{i}, S.event{i}, height(C), dur, ...
        std(ux), n_sig_freq(ux), Tx, Cx, sum(abs(ux)>opts.VThresh)*Tsm, ...
        std(uy), n_sig_freq(uy), Ty, Cy, sum(abs(uy)>opts.VThresh)*Tsm}; %#ok<AGROW>
end
rows = rows(~cellfun('isempty', rows));
E = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'log','label','event','n_rows','duration_s', ...
     'std_vx','nfreq_vx','Td0_vx','xcorr_vx','vx_ativo_s', ...
     'std_vy','nfreq_vy','Td0_vy','xcorr_vy','vy_ativo_s'});

fprintf('\n===== check_excitation_trans (ativo: |cmd|>%.2f m/s) =====\n', opts.VThresh);
labels = unique(E.label, 'stable');
for k = 1:numel(labels)
    Ek = E(strcmp(E.label, labels{k}), :);
    fprintf(['%-15s | %2d ev | vx: std med %.3f, ativo %4.1f s, Td0 %.3f, xc %.2f ' ...
        '| vy: std med %.3f, ativo %4.1f s, Td0 %.3f, xc %.2f\n'], labels{k}, height(Ek), ...
        median(Ek.std_vx), sum(Ek.vx_ativo_s), median(Ek.Td0_vx,'omitnan'), median(Ek.xcorr_vx,'omitnan'), ...
        median(Ek.std_vy), sum(Ek.vy_ativo_s), median(Ek.Td0_vy,'omitnan'), median(Ek.xcorr_vy,'omitnan'));
end
if opts.SaveCsv
    stamp = char(datetime('now','Format','yyyyMMdd_HHmm'));
    outcsv = fullfile('results', sprintf('check_excitation_trans_%s.csv', stamp));
    writetable(E, outcsv); fprintf('Salvo: %s\n', outcsv);
end
end

% =========================================================================
function n = n_sig_freq(u)
u = double(u(:)); u = u - mean(u); m = numel(u);
if m < 16 || all(u==0), n = 0; return; end
w = 0.5*(1 - cos(2*pi*(0:m-1)'/(m-1)));
P = abs(fft(u.*w)).^2; P = P(2:floor(m/2));
if isempty(P) || max(P)==0, n = 0; return; end
pk = (P(2:end-1) > P(1:end-2)) & (P(2:end-1) > P(3:end)) & (P(2:end-1) >= 0.1*max(P));
n = nnz(pk);
end

% =========================================================================
function [Td0, xcm] = xcorr_delay(u, y, Ts, max_lag_s)
K = min(round(max_lag_s/Ts), numel(u)-10);
r = -inf(K+1,1);
for k = 0:K
    a = u(1:end-k); b = y(1+k:end);
    if std(a)>0 && std(b)>0, r(k+1) = corr(a,b); end
end
[xcm, idx] = max(r); Td0 = (idx-1)*Ts;
if ~isfinite(xcm), Td0 = NaN; xcm = NaN; end
end
