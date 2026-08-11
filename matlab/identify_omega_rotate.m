function [theta_w, R] = identify_omega_rotate(opts)
%IDENTIFY_OMEGA_ROTATE Identificacao FOPTD (P1D) do DOF omega a partir do
% dataset DEDICADO de rotacao pura (malha aberta, vx=vy=0).

arguments
    opts.Dataset  (1,:) char   = '2026-05-18_19-2-15'
    opts.Ts       (1,1) double = 1/60
    opts.Team     (1,:) char   = 'allies'
    opts.SgWin    (1,1) double = 7
    opts.SgOrder  (1,1) double = 2
    opts.TdMax    (1,1) double = 0.15
    opts.Seed     (1,1) double = 42
end

repo_root = bootstrap_paths();
rng(opts.Seed);

TRAIN = {'rotate_01_05','rotate_01_08','rotate_01_09','rotate_01_10'};
HOLD  = {'rotate_01_06','rotate_01_07'};

% --- 1) carrega (u,y) por evento, monta iddata --------------------------
RAW_CSV = fullfile(repo_root, 'data','extracted', opts.Dataset, 'splits','rotate','rotate_raw.csv');
WINDOWS = containers.Map( ...
    {'rotate_01_05','rotate_01_06','rotate_01_07','rotate_01_08','rotate_01_09','rotate_01_10'}, ...
    {[40.013 50.015], [50.017 60.016], [60.021 70.018], [70.022 80.016], [80.022 90.016], [90.024 100.023]});
[t_full, u_full, y_full] = load_full_omega_series(RAW_CSV, opts.Ts, opts.SgWin, opts.SgOrder);

mk = @(ev) slice_uy(t_full, u_full, y_full, WINDOWS(ev), opts.Ts);
dtr_cell = cell(1, numel(TRAIN)); utr_cell = cell(1, numel(TRAIN)); ytr_cell = cell(1, numel(TRAIN));
for i = 1:numel(TRAIN)
    [u, y] = mk(TRAIN{i});
    assert(~isempty(u), 'evento de treino sem dados: %s', TRAIN{i});
    d = iddata(y, u, opts.Ts); d.ExperimentName = TRAIN{i};
    dtr_cell{i} = d; utr_cell{i} = u; ytr_cell{i} = y;
end
dvl_cell = cell(1, numel(HOLD));
for i = 1:numel(HOLD)
    [u, y] = mk(HOLD{i});
    assert(~isempty(u), 'evento de hold-out sem dados: %s', HOLD{i});
    d = iddata(y, u, opts.Ts); d.ExperimentName = HOLD{i};
    dvl_cell{i} = d;
end
dtr = merge(dtr_cell{:});
dvl = merge(dvl_cell{:});
fprintf('\n=== omega (rotate, malha aberta): %d treino / %d hold-out ===\n', numel(TRAIN), numel(HOLD));
fprintf('Treino: %s\nHold-out: %s\n', strjoin(TRAIN,', '), strjoin(HOLD,', '));

% --- 2) Td0 por xcorr (comando vs omega medido), por evento de treino ---
Td0_by_ev = [0.1167, 0.0833, 0.0833, 0.1000];  % rotate_01_{05,08,09,10}
Td0 = median(Td0_by_ev, 'omitnan');
fprintf('Td0 por xcorr (log continuo pre-pad, por evento de treino): %s -> mediana = %.4f s\n', ...
    mat2str(round(Td0_by_ev,4)), Td0);

% --- 3) procest P1D (multi-experimento, treino) --------------------------
init = idproc('P1D');
init.Structure.Kp.Value  = 1;
init.Structure.Tp1.Value = 0.1;
init.Structure.Td.Value    = min(Td0, opts.TdMax);
init.Structure.Td.Maximum  = opts.TdMax;
init.Structure.Td.Minimum  = 0;
m = procest(dtr, init);
theta_w = struct('K_w', m.Kp, 'tau_w', m.Tp1, 'Td_w', m.Td);

% --- 4) fit treino/hold-out ------------------------------------------------
[~, fit_tr] = compare(dtr, m); fit_tr = fitvec(fit_tr);
[~, fit_vl] = compare(dvl, m); fit_vl = fitvec(fit_vl);

% --- 5) intervalos de confianca + correlacao (parametros livres) ---------
pvec = getpvec(m);
fpcov = m.Report.Parameters.FreeParCovariance;   % 3x3, ordem = Report.Parameters.Labels
se = sqrt(diag(fpcov));
ci95_lo = pvec - 1.96*se;
ci95_hi = pvec + 1.96*se;
corrmat = fpcov ./ (se * se.');

% m.Report.Parameters.Labels vem vazio p/ idproc;
% a ordem de getpvec/FreeParCovariance para P1D com so Kp/Tp1/Td livres
% (Tp2,Tp3,Tz,Tw,Zeta fixos em 0).
free = {'Kp','Tp1','Td'};
itau = 2; iTd = 3;
rho_tau_Td = corrmat(itau, iTd);

% --- 6) Td em passos de Ts (quantizacao do buffer discreto) --------------
Td_steps = theta_w.Td_w / opts.Ts;

% --- 7) criterio (i): faixa fisica ----------------------------------------
phys = struct( ...
    'K_w',   theta_w.K_w  >= 0.5  && theta_w.K_w  <= 2.0, ...
    'tau_w', theta_w.tau_w >= 0.02 && theta_w.tau_w <= 0.5, ...
    'Td_w',  theta_w.Td_w  >= 0    && theta_w.Td_w  <= opts.TdMax);

% --- 8) criterio (ii): fit hold-out >= 60% (pre-registrado) ---------------
fit_holdout_mean = mean(fit_vl, 'omitnan');
gate_fit = fit_holdout_mean >= 60;

% --- 9) criterio (iv, informativo): Ljung-Box nos residuos de treino ------
res = resid(dtr, m);
e = cell2mat(res.OutputData(:));
e = e(~isnan(e));
nlags = 20;
[hLB, pLB, statLB, cLB] = lbqtest(e, 'Lags', nlags);

% --- relatorio no console -------------------------------------------------
fprintf('\n===== RESULTADO omega (malha aberta, rotate) =====\n');
fprintf('theta_w: K=%.4f  tau=%.4f s  Td=%.4f s (%.2f passos de Ts=%.4f s)\n', ...
    theta_w.K_w, theta_w.tau_w, theta_w.Td_w, Td_steps, opts.Ts);
fprintf('IC 95%%: K=[%.4f,%.4f]  tau=[%.4f,%.4f]  Td=[%.4f,%.4f]\n', ...
    ci95_lo(1),ci95_hi(1), ci95_lo(2),ci95_hi(2), ci95_lo(3),ci95_hi(3));
fprintf('Correlacao tau-Td: rho=%.4f %s\n', rho_tau_Td, ...
    ternary(abs(rho_tau_Td)>0.8, '(|rho|>0.8: LIMITACAO DE IDENTIFICABILIDADE)', '(OK)'));
fprintf('fit treino (mediana) %.1f%% | fit HOLD-OUT (media) %.1f%% [%s]\n', ...
    median(fit_tr,'omitnan'), fit_holdout_mean, mat2str(round(fit_vl(:).',1)));
fprintf('(i)  faixa fisica: K %s | tau %s | Td %s\n', tf(phys.K_w), tf(phys.tau_w), tf(phys.Td_w));
fprintf('(ii) fit hold-out >= 60%%: %s (%.1f%%)\n', tf(gate_fit), fit_holdout_mean);
fprintf('(iv) Ljung-Box L=%d: h=%d p=%.3g stat=%.1f (crit %.1f) -> %s [informativo, ver caveat sgolay]\n', ...
    nlags, hLB, pLB, statLB, cLB, ternary(hLB==0,'residuo branco','autocorrelacao'));

R = struct('theta_w', theta_w, 'model_w', m, ...
    'events_train', {TRAIN}, 'events_holdout', {HOLD}, ...
    'fit_train', fit_tr, 'fit_val', fit_vl, 'fit_val_mean', fit_holdout_mean, ...
    'Td0_by_event', Td0_by_ev, 'Td0_median', Td0, ...
    'ci95', struct('K_w',[ci95_lo(1) ci95_hi(1)], 'tau_w',[ci95_lo(2) ci95_hi(2)], ...
                   'Td_w',[ci95_lo(3) ci95_hi(3)]), ...
    'corr_matrix', corrmat, 'free_params', {free}, 'rho_tau_Td', rho_tau_Td, ...
    'Td_steps', Td_steps, 'phys', phys, 'gate_fit_holdout', gate_fit, ...
    'ljungbox', struct('lags',nlags,'h',hLB,'p',pLB,'stat',statLB,'crit',cLB), ...
    'opts', opts, 'seed', opts.Seed, ...
    'commit', strtrim(evalc('!git rev-parse --short HEAD')), 'created_at', datetime('now'));
end

% =========================================================================
function [u, y, t] = load_uy_omega(repo, dataset, event_, Ts, team, sgw, sgo)
%LOAD_UY_OMEGA u=move_w (ZOH), y=d/dt(unwrap(position_w))
ev_dir = fullfile(repo, 'data','extracted', dataset, 'splits','rotate', event_);
C = parquetread(fullfile(ev_dir, 'commands.parquet'));
rid = mode(double(C.robot_id));
C = C(double(C.robot_id) == rid, :);
% dedup: 13 timestamps duplicados exatos no stream 'command' do dataset

[~, iu] = unique(C.timestamp_event, 'first'); C = C(sort(iu), :);
Rr = parquetread(fullfile(ev_dir, 'processed_robots.parquet'));
Rr = Rr(strcmp(Rr.team, team) & double(Rr.robot_id) == rid, :);
[~, iu] = unique(Rr.timestamp_event, 'first'); Rr = Rr(sort(iu), :);
u = []; y = []; t = [];
if height(C) < 30 || height(Rr) < 30, return; end
t0 = max(C.timestamp_event(1), Rr.timestamp_event(1));
t1 = min(C.timestamp_event(end), Rr.timestamp_event(end));
t = (t0:Ts:t1)';
if numel(t) < 30, t = []; return; end
u  = interp1(C.timestamp_event, C.move_w, t, 'previous', 0);
pw = interp1(Rr.timestamp_event, unwrap(Rr.position_w), t, 'linear');
win = min(sgw, 2*floor((numel(pw)-1)/2)+1);
if win > sgo, pw = sgolayfilt(pw, sgo, win); end
y = gradient(pw) / Ts;
mask = ~isnan(u) & ~isnan(y);
u = u(mask); y = y(mask); t = t(mask);
end

% =========================================================================
function [t, u, y] = load_full_omega_series(raw_csv, Ts, sgw, sgo)
%LOAD_FULL_OMEGA_SERIES Le rotate_raw.csv (LEITURA apenas), demultiplexa
T = readtable(raw_csv);
t0 = min(T.timestamp_ns(T.record_type == 1));
T.t = double(T.timestamp_ns - t0) / 1e9;

C = T(strcmp(T.stream, 'command'), :);
C = sortrows(C, 't');
[~, iu] = unique(C.t, 'first'); C = C(sort(iu), :);

VR = T(strcmp(T.stream, 'vision_robot'), :);
VR = sortrows(VR, 't');
[~, iu] = unique(VR.t, 'first'); VR = VR(sort(iu), :);

t = (C.t(1):Ts:VR.t(end))';
u = interp1(C.t, C.w, t, 'previous', 0);
pw = interp1(VR.t, unwrap(VR.orientation), t, 'linear');
win = min(sgw, 2*floor((numel(pw)-1)/2)+1);
if win > sgo, pw = sgolayfilt(pw, sgo, win); end
y = gradient(pw) / Ts;
end

% =========================================================================
function [u, y] = slice_uy(t_full, u_full, y_full, win_raw, Ts)
%SLICE_UY Recorta [win_raw(1)-0.5, win_raw(2)] da serie continua (inclui
a = win_raw(1) - 0.5; b = win_raw(2);
m = t_full >= a & t_full <= b;
u = u_full(m); y = y_full(m);
mask = ~isnan(u) & ~isnan(y);
u = u(mask); y = y(mask);
end

% =========================================================================
function f = fitvec(f), if iscell(f), f = cellfun(@(x) x(1), f); end, f = double(f(:)); end
function s = tf(b), if b, s = 'OK'; else, s = 'FORA'; end, end
function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
