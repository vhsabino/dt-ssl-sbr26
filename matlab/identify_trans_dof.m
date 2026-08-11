function [theta_dof, R] = identify_trans_dof(dof, opts)
%IDENTIFY_TRANS_DOF Fase 2 — identifica G_dof(s)=K e^{-Td s}/(tau s+1) para
% um DOF translacional (dof='x' ou 'y'), estrutura P1D (mesma de omega).
%

arguments
    dof (1,1) char {mustBeMember(dof,{'x','y'})}
    opts.Label    (1,:) char   = ''
    opts.Ts       (1,1) double = 1/60
    opts.Team     (1,:) char   = 'allies'
    opts.SgWin    (1,1) double = 7
    opts.SgOrder  (1,1) double = 2
    opts.FracHold (1,1) double = 1/3
    opts.Seed     (1,1) double = 42
    opts.TdInit   (1,1) double = 0.15
    opts.TdMax    (1,1) double = 0.15
end

repo_root = bootstrap_paths();
rng(opts.Seed);
if isempty(opts.Label)
    if dof=='x', opts.Label = 'front_to_back'; else, opts.Label = 'side_to_side'; end
end
cmdcol = ['move_' dof];

% --- lista eventos do label ---
data_root = fullfile(repo_root,'data','extracted');
S = list_splits(data_root, 'Label', opts.Label);
S = S(strcmp(S.chunk,'commands'), :);
ev = struct('log',{},'event',{},'d',{});
for i = 1:height(S)
    [u, y, ~] = load_uy_trans(repo_root, S.log{i}, S.label{i}, S.event{i}, ...
        cmdcol, dof, opts.Ts, opts.Team, opts.SgWin, opts.SgOrder);
    if isempty(u) || numel(u) < 30, continue; end
    d = iddata(y, u, opts.Ts); d.ExperimentName = S.event{i};
    ev(end+1) = struct('log',S.log{i},'event',S.event{i},'d',d); %#ok<AGROW>
end
nq = numel(ev);
fprintf('\n=== DOF %s (label %s): %d eventos ===\n', dof, opts.Label, nq);

% --- hold-out por evento ---
nho = max(1, round(opts.FracHold*nq));
idx = randperm(nq); iva = sort(idx(1:nho)); itr = sort(idx(nho+1:end));
fprintf('Treino: %d | Hold-out: %d {%s}\n', numel(itr), numel(iva), ...
    strjoin({ev(iva).event}, ', '));
dtr = merge(ev(itr).d); dvl = merge(ev(iva).d);

% --- procest P1D ---
init = idproc('P1D');
init.Structure.Kp.Value = 1;  init.Structure.Tp1.Value = 0.1;
init.Structure.Td.Value = min(opts.TdInit,opts.TdMax);
init.Structure.Td.Minimum = 0; init.Structure.Td.Maximum = opts.TdMax;
m = procest(dtr, init);
theta_dof = struct('K', m.Kp, 'tau', m.Tp1, 'Td', m.Td);

[~, fit_tr] = compare(dtr, m); fit_tr = fitvec(fit_tr);
[~, fit_vl] = compare(dvl, m); fit_vl = fitvec(fit_vl);

% --- antigo (sysid_initial) p/ comparacao ---
th0 = load_theta_mat(fullfile('results','sysid_initial.mat'));
old = struct('K', th0.(['K_v' dof]), 'tau', th0.(['tau_v' dof]), 'Td', th0.(['Td_v' dof]));

phys = struct('K', theta_dof.K>=0.5 && theta_dof.K<=2.0, ...
    'tau', theta_dof.tau>=0.02 && theta_dof.tau<=0.5, ...
    'Td', theta_dof.Td>=0 && theta_dof.Td<=opts.TdMax);

fprintf('ANTIGO  v%s: K=%.4f tau=%.4f Td=%.4f\n', dof, old.K, old.tau, old.Td);
fprintf('NOVO    v%s: K=%.4f tau=%.4f Td=%.4f\n', dof, theta_dof.K, theta_dof.tau, theta_dof.Td);
fprintf('fit treino (mediana) %.1f%% | fit VAL (media) %.1f%% [%s]\n', ...
    median(fit_tr,'omitnan'), mean(fit_vl,'omitnan'), mat2str(round(fit_vl(:).',1)));
fprintf('(i) faixa fisica: K %s | tau %s | Td %s\n', tf(phys.K), tf(phys.tau), tf(phys.Td));

R = struct('dof',dof,'label',opts.Label,'model',m,'theta_new',theta_dof, ...
    'theta_old',old,'events_train',{{ev(itr).event}},'events_holdout',{{ev(iva).event}}, ...
    'fit_train',fit_tr,'fit_val',fit_vl,'fit_val_mean',mean(fit_vl,'omitnan'), ...
    'phys',phys,'seed',opts.Seed,'opts',opts, ...
    'commit',strtrim(evalc('!git rev-parse --short HEAD')),'created_at',datetime('now'));
end

% =========================================================================
function [u, y, t] = load_uy_trans(repo, log_, label_, event_, cmdcol, dof, Ts, team, sgw, sgo)
ev_dir = fullfile(repo,'data','extracted',log_,'splits',label_,event_);
C = parquetread(fullfile(ev_dir,'commands.parquet'));
rid = mode(double(C.robot_id)); C = C(double(C.robot_id)==rid,:);
R = parquetread(fullfile(ev_dir,'processed_robots.parquet'));
R = R(strcmp(R.team,team) & double(R.robot_id)==rid,:);
u=[]; y=[]; t=[];
if height(C)<30 || height(R)<30, return; end
t0=max(C.timestamp_event(1),R.timestamp_event(1));
t1=min(C.timestamp_event(end),R.timestamp_event(end));
t=(t0:Ts:t1)'; if numel(t)<30, t=[]; return; end
u  = interp1(C.timestamp_event, C.(cmdcol), t, 'previous', 0);
px = interp1(R.timestamp_event, R.position_x/1000, t, 'linear');
py = interp1(R.timestamp_event, R.position_y/1000, t, 'linear');
th = interp1(R.timestamp_event, unwrap(R.position_w), t, 'linear');
win = min(sgw, 2*floor((numel(t)-1)/2)+1);
if win>sgo, px=sgolayfilt(px,sgo,win); py=sgolayfilt(py,sgo,win); end
vxw = gradient(px)/Ts; vyw = gradient(py)/Ts;
if dof=='x', y =  cos(th).*vxw + sin(th).*vyw;
else,        y = -sin(th).*vxw + cos(th).*vyw; end
m = ~isnan(u) & ~isnan(y); u=u(m); y=y(m); t=t(m);
end

% =========================================================================
function f = fitvec(f), if iscell(f), f=cellfun(@(x)x(1),f); end, f=double(f(:)); end
function s = tf(b), if b, s='OK'; else, s='FORA'; end, end
