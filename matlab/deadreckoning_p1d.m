function [xs, ys, ths, vbody] = deadreckoning_p1d(s, theta)
%DEADRECKONING_P1D Dead-reckoning analitico do gemeo com dinamica P1D por DOF.
% Generaliza ideal_deadreckoning (que e o caso K=1, tau=0, Td=0) aplicando
% G_i(s) = K_i * e^{-Td_i s} / (tau_i s + 1) a cada comando de corpo antes da
% integracao em frame global — exatamente a cadeia do modelo Simulink pid_v3
% (TransferFcn -> TransportDelay -> rodas[identidade, dz=0] -> rotacao ->
% integradores). A reconciliacao analitico-vs-Simulink fecha em <6 mm para o
% caso ideal (ver relatorio §4); este arquivo e validado contra a trajetoria
% Simulink salva do placeholder antes de usar (NaN-safe).
%
% s    : struct de load_split (cmd_*_ts na grade s.t, pos0 com x0/y0/theta0).
% theta: struct com os 9 campos (load_theta_mat); ordem fixa do SLDD.
% Saidas xs,ys,ths na grade s.t; vbody = [vx vy vw] de corpo pos-G(s).

arguments
    s     (1,1) struct
    theta (1,1) struct
end

t  = s.t;
ux = series_on_grid(s.cmd_vx_ts, t);
uy = series_on_grid(s.cmd_vy_ts, t);
uw = series_on_grid(s.cmd_vw_ts, t);

vx = p1d_response(ux, t, theta.K_vx, theta.tau_vx, theta.Td_vx);
vy = p1d_response(uy, t, theta.K_vy, theta.tau_vy, theta.Td_vy);
vw = p1d_response(uw, t, theta.K_w,  theta.tau_w,  theta.Td_w);

ths = s.pos0.theta0 + cumtrapz(t, vw);
xs  = s.pos0.x0 + cumtrapz(t, cos(ths).*vx - sin(ths).*vy);
ys  = s.pos0.y0 + cumtrapz(t, sin(ths).*vx + cos(ths).*vy);
vbody = [vx, vy, vw];
end

% =========================================================================
function v = p1d_response(u, t, K, tau, Td)
% Atraso de transporte Td (ZOH, saida inicial 0) -> lag 1a ordem tau -> ganho K.
% LTI escalar: atraso e lag comutam; ordem irrelevante para o sinal de saida.
if Td > 0
    ud = interp1(t, u, t - Td, 'previous', 0);   % comando Td s atras (0 antes de t0)
    ud(isnan(ud)) = 0;
else
    ud = u;
end
if tau > 0
    v = lag_filter(ud, t, tau);
else
    v = ud;
end
v = K * v(:);
end

% =========================================================================
function u = series_on_grid(ts, t)
% Identico a ideal_deadreckoning: ZOH na grade do split.
if numel(ts.Time) == numel(t) && all(ts.Time == t)
    u = ts.Data;
else
    u = interp1(ts.Time, ts.Data, t, 'previous', 0);
end
u = u(:);
end

% =========================================================================
function v = lag_filter(u, t, tau)
% 1/(tau*s+1) discretizado exato sob ZOH no comando, estado inicial nulo
% (identico a ideal_deadreckoning).
v = zeros(size(u));
for k = 2:numel(u)
    a = 1 - exp(-(t(k) - t(k-1)) / tau);
    v(k) = v(k-1) + a * (u(k-1) - v(k-1));
end
end
