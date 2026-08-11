function [W, T, traj_file] = eval_omega_v3_se2(opts)
%EVAL_OMEGA_V3_SE2 Etapa C, criterio (iii): custo SE(2) multi-horizonte do
% omega LINEAR candidato vs placeholder nos eventos ricos em rotacao
% (shoot_to_goal), com Wilcoxon pareado por horizonte.
%
% Fidelidade: simula apenas o candidato novo pelo mesmo pipeline Simulink
% (run_one_mat) dos baselines salvos; o placeholder e REUSADO das trajetorias
% Simulink da campanha (results/traj_eval_campaign_*.mat) — nao se re-simula a
% campanha. Metrica = rmse_pos_rebased_se2 (re-ancora SE(2), theta truth
% filtrado), identica a usada na campanha. Pareamento por evento isola o
% efeito de trocar so o modelo de omega.
%
%  Salva CSV (por evento x horizonte), CSV de Wilcoxon e as
% trajetorias do candidato.

arguments
    opts.CandMat   (1,:) char = fullfile('build','sysid_omega_v3_linear_tmp.mat')
    opts.Campaign  (1,:) char = fullfile('results','traj_eval_campaign_20260611_2238.mat')
    opts.Model     (1,:) char = 'parameter_estimation_model_pid_v3'
    opts.Label     (1,:) char = 'shoot_to_goal'
    opts.Horizons  (1,:) double = [0.5 1 2 2.5 4]
    opts.Overlap   (1,1) double = 0.5
    opts.Seed      (1,1) double = 42
    opts.Dataset   (1,:) char = '2026-05-18_19-2-15'
end

repo_root = bootstrap_paths();
rng(opts.Seed);
Rc = load(opts.CandMat);                     % theta0, theta_w, R
holdout = string(Rc.R.events_holdout);
train   = string(Rc.R.events_train);

C = load(opts.Campaign);  traj = C.traj;
isP = strcmp({traj.theta},'procest_placeholder') & strcmp({traj.label},opts.Label);
idxP = find(isP);
fprintf('Eventos %s na campanha (placeholder): %d\n', opts.Label, numel(idxP));

% --- Simulink pronto (SLDD pode estar stale; restaura no fim) --------------
[~, githash] = system(sprintf('git -C "%s" rev-parse --short HEAD', repo_root));
githash = strtrim(githash);
Simulink.data.dictionary.closeAll('-discard');
ensure_model_ready(opts.Model);
sldd_restore = onCleanup(@() system(sprintf( ...
    'git -C "%s" restore config/ssl_robot.sldd', repo_root))); %#ok<NASGU>

rows = {};
traj_c = struct('label',{},'event',{},'t',{},'x_sim',{},'y_sim',{},'th_sim',{});
for j = 1:numel(idxP)
    tr = traj(idxP(j));   % placeholder (Simulink salvo) + truth
    ev = tr.event;
    role = "nao_qualif";
    if ismember(string(ev), holdout), role = "holdout";
    elseif ismember(string(ev), train), role = "treino"; end
    try
        cmd = fullfile(repo_root,'data','extracted',opts.Dataset,'splits', ...
            opts.Label, ev, 'commands.parquet');
        canon = canonical_split(cmd);
        out = run_one_mat(opts.Model, canon, opts.CandMat);
        [tsim, px, py] = extract_sim_position(out);
        pth = out.get('theta_aligned');
        xc = interp1(tsim, px, tr.t, 'linear', 'extrap');
        yc = interp1(tsim, py, tr.t, 'linear', 'extrap');
        thc= interp1(pth.Time, squeeze(pth.Data), tr.t, 'linear', 'extrap');
        traj_c(end+1) = struct('label',opts.Label,'event',ev,'t',tr.t, ...
            'x_sim',xc,'y_sim',yc,'th_sim',thc); %#ok<AGROW>
        for h = opts.Horizons
            rc = rmse_pos_rebased_se2(tr.t, xc,yc,thc, ...
                tr.x_truth,tr.y_truth,tr.th_truth, h, opts.Overlap);
            rp = rmse_pos_rebased_se2(tr.t, tr.x_sim,tr.y_sim,tr.th_sim, ...
                tr.x_truth,tr.y_truth,tr.th_truth, h, opts.Overlap);
            rows(end+1,:) = {ev, char(role), h, rc, rp, rc-rp}; %#ok<AGROW>
        end
        fprintf('  %-20s [%-9s] ok\n', ev, role);
    catch ME
        msg = regexprep(strrep(ME.message,newline,' '),'<[^>]+>','');
        fprintf('  %-20s ERRO: %s\n', ev, msg);
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'event','role','horizonte_s','se2_cand_m','se2_plac_m','delta_m'});

% --- Wilcoxon pareado por horizonte (candidato vs placeholder) -------------
wr = {};
for h = opts.Horizons
    sub = T(T.horizonte_s==h, :);
    rc = sub.se2_cand_m; rp = sub.se2_plac_m;
    ok = ~isnan(rc) & ~isnan(rp);
    p = signrank(rc(ok), rp(ok));
    wr(end+1,:) = {h, nnz(ok), median(rc(ok)), median(rp(ok)), ...
        median(rc(ok)-rp(ok)), p}; %#ok<AGROW>
end
% pooled (todos horizontes)
okall = ~isnan(T.se2_cand_m) & ~isnan(T.se2_plac_m);
wr(end+1,:) = {NaN, nnz(okall), median(T.se2_cand_m(okall)), ...
    median(T.se2_plac_m(okall)), median(T.delta_m(okall)), ...
    signrank(T.se2_cand_m(okall), T.se2_plac_m(okall))};
W = cell2table(wr, 'VariableNames', ...
    {'horizonte_s','n','median_cand_m','median_plac_m','median_delta_m','p_wilcoxon'});

fprintf('\n===== (iii) SE(2) multi-horizonte: candidato vs placeholder =====\n');
fprintf('%-8s %4s %12s %12s %12s %10s\n','h(s)','n','med_cand','med_plac','med_delta','p_wilcox');
for k = 1:height(W)
    hl = W.horizonte_s(k); if isnan(hl), hs='pool'; else, hs=sprintf('%.1f',hl); end
    fprintf('%-8s %4d %12.4f %12.4f %+12.4f %10.3g\n', hs, W.n(k), ...
        W.median_cand_m(k), W.median_plac_m(k), W.median_delta_m(k), W.p_wilcoxon(k));
end
fprintf('(delta>0 = candidato PIOR que placeholder)\n');

% --- artefatos -------------------------------------------------------------
stamp = char(datetime('now','Format','yyyyMMdd_HHmm'));
csv  = fullfile('results', sprintf('eval_omega_v3_se2_%s.csv', stamp));
wcsv = fullfile('results', sprintf('eval_omega_v3_wilcoxon_%s.csv', stamp));
writetable(T, csv); writetable(W, wcsv);
traj_file = fullfile('results', sprintf('traj_eval_omega_v3_%s.mat', stamp));
meta = struct('script','automation/eval_omega_v3_se2.m','commit',githash, ...
    'seed',opts.Seed,'overlap',opts.Overlap,'model',opts.Model, ...
    'cand_mat',opts.CandMat,'campaign',opts.Campaign, ...
    'theta_w',Rc.theta_w,'generated',char(datetime('now')));
save(traj_file, 'traj_c', 'T', 'W', 'meta', '-v7.3');
fprintf('Salvo: %s\n        %s\n        %s\n', csv, wcsv, traj_file);
end
