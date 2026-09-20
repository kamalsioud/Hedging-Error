import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, differential_evolution


def analytical_price(S0, K, T, r, q, sigma):
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def monte_carlo_call(S0, K, T, r, q, sigma, n_paths, seed):
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(n_paths)
    ST  = S0*np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    discounted = np.exp(-r*T)*np.maximum(ST - K, 0)
    return discounted.mean(), discounted.std(ddof=1)/np.sqrt(n_paths)


def bs_price(S0, K, T, r, q, vol, kind):
    d1 = (np.log(S0/K) + (r - q + 0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    if kind == 'c':
        return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    if kind == 'p':
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1)


def implied_vol(price, S0, K, T, r, q, kind):
    try:
        return brentq(lambda vol: bs_price(S0, K, T, r, q, vol, kind) - price,
                      0.0001, 3.0)
    except ValueError:
        return np.nan


def bs_delta(S0, K, T, r, q, vol):
    d1 = (np.log(S0/K) + (r - q + 0.5*vol**2)*T)/(vol*np.sqrt(T))
    return np.exp(-q*T)*norm.cdf(d1)


def bs_vega(S0, K, T, r, q, vol):
    d1 = (np.log(S0/K) + (r - q + 0.5*vol**2)*T)/(vol*np.sqrt(T))
    return S0*np.exp(-q*T)*np.sqrt(T)*norm.pdf(d1)


def fd_delta(S0, K, T, r, q, vol, eps):
    up   = bs_price(S0 + eps, K, T, r, q, vol, 'c')
    down = bs_price(S0 - eps, K, T, r, q, vol, 'c')
    return (up - down) / (2*eps)


def heston_cf(xi, j, S0, T, r, q, v0, kappa, theta, sigma, rho):
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho*sigma if j == 1 else kappa
    d = np.sqrt((rho*sigma*1j*xi - b)**2 - sigma**2*(2*u*1j*xi - xi**2))
    g = (b - rho*sigma*1j*xi - d) / (b - rho*sigma*1j*xi + d)

    C = ((r - q)*1j*xi*T + (kappa*theta/sigma**2)
         * ((b - rho*sigma*1j*xi - d)*T
            - 2*np.log((1 - g*np.exp(-d*T))/(1 - g))))
    D = ((b - rho*sigma*1j*xi - d)/sigma**2) * ((1 - np.exp(-d*T))
                                                / (1 - g*np.exp(-d*T)))
    return np.exp(C + D*v0 + 1j*xi*np.log(S0))


def heston_call(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                n_nodes=1200, xi_max=350.0):
    xi = np.linspace(1e-8, xi_max, n_nodes)
    P  = []
    for j in (1, 2):
        f = heston_cf(xi, j, S0, T, r, q, v0, kappa, theta, sigma, rho)
        integrand = np.real(np.exp(-1j*xi*np.log(K)) * f / (1j*xi))
        P.append(0.5 + np.trapezoid(integrand, xi)/np.pi)
    return S0*np.exp(-q*T)*P[0] - K*np.exp(-r*T)*P[1]


def heston_delta(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                 n_nodes=1200, xi_max=350.0):
    xi = np.linspace(1e-8, xi_max, n_nodes)
    f  = heston_cf(xi, 1, S0, T, r, q, v0, kappa, theta, sigma, rho)
    integrand = np.real(np.exp(-1j*xi*np.log(K)) * f / (1j*xi))
    P1 = 0.5 + np.trapezoid(integrand, xi)/np.pi
    return np.exp(-q*T)*P1


def heston_vega(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                n_nodes=10000, xi_max=3000.0):
    xi = np.linspace(1e-8, xi_max, n_nodes)
    dP = []
    for j in (1, 2):
        u = 0.5 if j == 1 else -0.5
        b = kappa - rho*sigma if j == 1 else kappa
        d = np.sqrt((rho*sigma*1j*xi - b)**2 - sigma**2*(2*u*1j*xi - xi**2))
        g = (b - rho*sigma*1j*xi - d) / (b - rho*sigma*1j*xi + d)
        D = ((b - rho*sigma*1j*xi - d)/sigma**2) * ((1 - np.exp(-d*T))
                                                    / (1 - g*np.exp(-d*T)))
        f = heston_cf(xi, j, S0, T, r, q, v0, kappa, theta, sigma, rho)
        integrand = np.real(np.exp(-1j*xi*np.log(K)) * D * f / (1j*xi))
        dP.append(np.trapezoid(integrand, xi)/np.pi)
    return S0*np.exp(-q*T)*dP[0] - K*np.exp(-r*T)*dP[1]


def heston_euler(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                 n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    S   = np.full(n_paths, S0)
    v   = np.full(n_paths, v0)

    for _ in range(n_steps):
        Zv = rng.standard_normal(n_paths)
        Zp = rng.standard_normal(n_paths)
        Zs = rho*Zv + np.sqrt(1 - rho**2)*Zp

        v = np.maximum(v + kappa*(theta - v)*dt
                         + sigma*np.sqrt(v)*np.sqrt(dt)*Zv, 0)

        S = S * np.exp((r - q - 0.5*v)*dt
                       + np.sqrt(v)*np.sqrt(dt)*Zs)

    discounted = np.exp(-r*T)*np.maximum(S - K, 0)
    return discounted.mean(), discounted.std(ddof=1)/np.sqrt(n_paths)


def heston_qe(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
              n_paths, n_steps, seed, psi_c=1.5):
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    E   = np.exp(-kappa*dt)
    g1  = g2 = 0.5

    K0 = -rho*kappa*theta*dt/sigma
    K1 = g1*dt*(kappa*rho/sigma - 0.5) - rho/sigma
    K2 = g2*dt*(kappa*rho/sigma - 0.5) + rho/sigma
    K3 = g1*dt*(1 - rho**2)
    K4 = g2*dt*(1 - rho**2)

    logS = np.full(n_paths, np.log(S0))
    v    = np.full(n_paths, v0)

    for _ in range(n_steps):
        m   = theta + (v - theta)*E
        s2  = (v*sigma**2*E*(1 - E)/kappa
               + theta*sigma**2*(1 - E)**2/(2*kappa))
        psi = s2 / m**2

        Zv = rng.standard_normal(n_paths)
        U  = rng.random(n_paths)
        Z  = rng.standard_normal(n_paths)
        v_new = np.empty(n_paths)

        quad = psi <= psi_c
        inv  = 2 / psi[quad]
        b2   = inv - 1 + np.sqrt(inv*(inv - 1))
        a    = m[quad] / (1 + b2)
        v_new[quad] = a * (np.sqrt(b2) + Zv[quad])**2

        expo = ~quad
        p    = (psi[expo] - 1) / (psi[expo] + 1)
        beta = (1 - p) / m[expo]
        v_new[expo] = np.where(U[expo] <= p, 0.0,
                               np.log((1 - p)/(1 - U[expo])) / beta)

        logS += ((r - q)*dt + K0 + K1*v + K2*v_new
                 + np.sqrt(np.maximum(K3*v + K4*v_new, 0)) * Z)
        v = v_new

    discounted = np.exp(-r*T)*np.maximum(np.exp(logS) - K, 0)
    return discounted.mean(), discounted.std(ddof=1)/np.sqrt(n_paths)


def simulate_market(S0, T, r, q, v0, kappa, theta, sigma, rho,
                    n_paths, n_steps, seed, psi_c=1.5):
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    E   = np.exp(-kappa*dt)
    g1  = g2 = 0.5

    K0 = -rho*kappa*theta*dt/sigma
    K1 = g1*dt*(kappa*rho/sigma - 0.5) - rho/sigma
    K2 = g2*dt*(kappa*rho/sigma - 0.5) + rho/sigma
    K3 = g1*dt*(1 - rho**2)
    K4 = g2*dt*(1 - rho**2)

    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))
    logS = np.full(n_paths, np.log(S0))
    vt   = np.full(n_paths, v0)
    S[0], v[0] = S0, v0

    for i in range(1, n_steps + 1):
        m   = theta + (vt - theta)*E
        s2  = (vt*sigma**2*E*(1 - E)/kappa
               + theta*sigma**2*(1 - E)**2/(2*kappa))
        psi = s2 / m**2

        Zv = rng.standard_normal(n_paths)
        U  = rng.random(n_paths)
        Z  = rng.standard_normal(n_paths)
        v_new = np.empty(n_paths)

        quad = psi <= psi_c
        inv  = 2 / psi[quad]
        b2   = inv - 1 + np.sqrt(inv*(inv - 1))
        v_new[quad] = (m[quad]/(1 + b2)) * (np.sqrt(b2) + Zv[quad])**2

        expo = ~quad
        p    = (psi[expo] - 1) / (psi[expo] + 1)
        beta = (1 - p) / m[expo]
        v_new[expo] = np.where(U[expo] <= p, 0.0,
                               np.log((1 - p)/(1 - U[expo])) / beta)

        logS += ((r - q)*dt + K0 + K1*vt + K2*v_new
                 + np.sqrt(np.maximum(K3*vt + K4*v_new, 0)) * Z)
        vt = v_new
        S[i], v[i] = np.exp(logS), vt

    return S, v


def qe_terminal(S_t, v_t, tau, n_inner, rng, n_steps=20, psi_c=1.5):
    S_t = np.atleast_1d(np.asarray(S_t, float))
    v_t = np.atleast_1d(np.asarray(v_t, float))
    dt  = tau / n_steps
    E   = np.exp(-kappa*dt)
    g1  = g2 = 0.5
    K0 = -rho*kappa*theta*dt/sigma
    K1 = g1*dt*(kappa*rho/sigma - 0.5) - rho/sigma
    K2 = g2*dt*(kappa*rho/sigma - 0.5) + rho/sigma
    K3 = g1*dt*(1 - rho**2)
    K4 = g2*dt*(1 - rho**2)

    shape = (S_t.size, n_inner)
    logS  = np.log(S_t)[:, None] * np.ones(shape)
    v     = v_t[:, None] * np.ones(shape)

    for _ in range(n_steps):
        m   = theta + (v - theta)*E
        s2  = (v*sigma**2*E*(1 - E)/kappa
               + theta*sigma**2*(1 - E)**2/(2*kappa))
        psi = s2 / m**2

        Zv = rng.standard_normal(shape)
        U  = rng.random(shape)
        Z  = rng.standard_normal(shape)
        v_new = np.empty(shape)

        quad = psi <= psi_c
        inv  = 2 / psi[quad]
        b2   = inv - 1 + np.sqrt(inv*(inv - 1))
        v_new[quad] = (m[quad]/(1 + b2)) * (np.sqrt(b2) + Zv[quad])**2

        expo = ~quad
        p    = (psi[expo] - 1) / (psi[expo] + 1)
        beta = (1 - p) / m[expo]
        v_new[expo] = np.where(U[expo] <= p, 0.0,
                               np.log((1 - p)/(1 - U[expo])) / beta)

        logS += ((r - q)*dt + K0 + K1*v + K2*v_new
                 + np.sqrt(np.maximum(K3*v + K4*v_new, 0)) * Z)
        v = v_new

    return np.exp(logS)


def delta_mc(S_t, v_t, tau, K, seed):
    rng = np.random.default_rng(seed)
    ST  = qe_terminal(S_t, v_t, tau, n_inner, rng)

    up   = np.exp(-r*tau)*np.maximum((1 + eps_S)*ST - K, 0).mean(1)
    down = np.exp(-r*tau)*np.maximum((1 - eps_S)*ST - K, 0).mean(1)
    return (up - down) / (2*eps_S*S_t)


def vega_mc(S_t, v_t, tau, K, seed, eps):
    rng   = np.random.default_rng(seed)
    ST_up = qe_terminal(S_t, v_t + eps, tau, n_inner, rng)

    rng   = np.random.default_rng(seed)
    ST_dn = qe_terminal(S_t, np.maximum(v_t - eps, 0.0), tau, n_inner, rng)

    up = np.exp(-r*tau)*np.maximum(ST_up - K, 0).mean(1)
    dn = np.exp(-r*tau)*np.maximum(ST_dn - K, 0).mean(1)

    span = (v_t + eps) - np.maximum(v_t - eps, 0.0)
    return (up - dn) / span


def mc_greeks(S_t, v_t, tau, K, seed):
    return (delta_mc(S_t, v_t, tau, K, seed),
            vega_mc(S_t, v_t, tau, K, seed, eps_v))


def mae_es(pnl, alpha=0.05):
    mae = np.abs(pnl).mean()
    es  = pnl[pnl <= np.quantile(pnl, alpha)].mean()
    return mae, es


def paired_bootstrap(a, b, stat, n_boot=4000, seed=0):
    rng  = np.random.default_rng(seed)
    n    = len(a)
    obs  = stat(a) - stat(b)
    draw = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        draw[k] = stat(a[idx]) - stat(b[idx])
    se = draw.std(ddof=1)
    z  = obs/se if se > 0 else 0.0
    return obs, se, 2*(1 - norm.cdf(abs(z)))


spx = pd.read_csv('spx.csv', header=None, names=['date', 'price'],
                  index_col=0, parse_dates=True, dayfirst=True)

spx_logret = np.log(spx['price'] / spx['price'].shift(1)).dropna()

plt.figure(figsize=(12, 4))
plt.plot(spx_logret, linewidth=0.6, color='black')
plt.ylabel('Daily return')
plt.show()


print(analytical_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.2))
for n in (1000, 10000, 100000, 1000000):
    print(n, monte_carlo_call(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, n, seed=1))


S0 = 7791.76

info = {'august.csv':    (8,   0.0401, 0.0136),
        'September.csv': (36,  0.0401, 0.0085),
        'October.csv':   (64,  0.0400, 0.0049),
        'november.csv':  (99,  0.0402, 0.0050),
        'december.csv':  (127, 0.0405, 0.0055)}

frames = []
for f, header in info.items():
    df = pd.read_csv(f, skiprows=3, header=None, usecols=[0, 2, 3, 9, 10],
                     names=['strike', 'c_bid', 'c_ask', 'p_bid', 'p_ask'])
    df['T'], df['r'], df['q'] = header[0]/365, header[1], header[2]
    frames.append(df)

chain = pd.concat(frames, ignore_index=True)

chain['kind'] = np.where(chain.strike < S0, 'p', 'c')
chain['bid']  = np.where(chain.strike < S0, chain.p_bid, chain.c_bid)
chain['ask']  = np.where(chain.strike < S0, chain.p_ask, chain.c_ask)

chain = chain[chain.bid > 0]
chain['mid'] = (chain.bid + chain.ask) / 2
chain = chain[(chain.strike >= 0.80*S0) & (chain.strike <= 1.20*S0)]

chain['market_iv'] = [implied_vol(m, S0, k, t, r_c, q_c, kd)
                      for m, k, t, r_c, q_c, kd
                      in zip(chain.mid, chain.strike, chain['T'],
                             chain.r, chain.q, chain.kind)]
chain = chain.dropna(subset=['market_iv'])

chain = chain[chain['T'] > 8/365]
print(len(chain))


bounds = [(0.001, 0.25),
          (0.1,   15),
          (0.001, 0.25),
          (0.01,  3),
          (-0.99, -0.01)]


def error(x):
    v0, kappa, theta, sigma, rho = x
    model_iv = []
    for K, T, r_c, q_c, kind in zip(chain.strike, chain['T'],
                                    chain.r, chain.q, chain.kind):
        c = heston_call(S0, K, T, r_c, q_c, v0, kappa, theta, sigma, rho)
        price = c if kind == 'c' else c - S0*np.exp(-q_c*T) + K*np.exp(-r_c*T)
        model_iv.append(implied_vol(price, S0, K, T, r_c, q_c, kind))
    return np.nansum((np.array(model_iv) - chain.market_iv.values)**2)


result = differential_evolution(error, bounds, seed=42,
                                maxiter=60, popsize=15, workers=-1)
v0, kappa, theta, sigma, rho = result.x
print(v0, kappa, theta, sigma, rho)


model_iv = []
for K, T, r, q, kind in zip(chain.strike, chain['T'],
                            chain.r, chain.q, chain.kind):
    c = heston_call(S0, K, T, r, q, v0, kappa, theta, sigma, rho)
    price = c if kind == 'c' else c - S0*np.exp(-q*T) + K*np.exp(-r*T)
    model_iv.append(implied_vol(price, S0, K, T, r, q, kind))

chain['model_iv'] = model_iv
chain['err']      = 100*(chain.market_iv - chain.model_iv)/chain.market_iv
chain['expiry']   = chain['T']*365

print(chain.groupby('expiry').err.agg(
    mean='mean',
    mean_abs=lambda e: e.abs().mean()))

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(chain.strike, chain.expiry, 100*chain.market_iv,
                cmap='viridis', linewidth=0, alpha=0.55)
ax.plot_trisurf(chain.strike, chain.expiry, 100*chain.model_iv,
                cmap='plasma', linewidth=0, alpha=0.55)
ax.set_xlabel('Strike'); ax.set_ylabel('Maturity (days)')
ax.set_zlabel('Implied volatility (%)')
ax.view_init(elev=25, azim=-125)
plt.tight_layout(); plt.show()


S0 = 7791.76
v0, kappa, theta, sigma, rho = 0.01392, 3.730, 0.05663, 1.201, -0.596
print(2*kappa*theta, sigma**2)


K, T, r, q = 8800.0, 36/365, 0.0401, 0.0085

for xi_max in (50, 100, 150, 250, 300, 350, 400, 600):
    print(xi_max, heston_call(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                              n_nodes=1200, xi_max=xi_max))

for xi_max in (350, 600, 1000, 1500, 3000, 6000, 12000):
    print(xi_max, heston_vega(S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                              n_nodes=10000, xi_max=xi_max))


K, T, r, q = 7800.0, 127/365, 0.0405, 0.0055
K_V, K_C   = 7800.0, 8200.0

iv   = 0.1518
iv_V = 0.1518
iv_C = None

print(heston_call(S0, K, T, r, q, v0, kappa, theta, sigma, rho))
print(heston_delta(S0, K, T, r, q, v0, kappa, theta, sigma, rho))
print(heston_vega(S0, K, T, r, q, v0, kappa, theta, sigma, rho))
print(bs_delta(S0, K, T, r, q, iv))
print(bs_vega(S0, K_V, T, r, q, iv_V))


for n in (10000, 100000, 1000000):
    print(n,
          heston_euler(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n, 127, 1),
          heston_qe(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n, 127, 1))


for eps in (10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001):
    print(eps, fd_delta(S0, K, T, r, q, iv, eps))


n_paths = 1000
n_rebal = 127
dt      = T / n_rebal
eps_S   = 0.0001
eps_v   = v0
n_inner = 3000

S, v = simulate_market(S0, T, r, q, v0, kappa, theta, sigma, rho,
                       n_paths, n_rebal, seed=20240101)


truth = heston_vega(S0, K_V, T, r, q, v0, kappa, theta, sigma, rho)

for mult in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16):
    eps  = mult*v0
    est  = np.array([vega_mc(np.array([S0]), np.array([v0]), T, K_V,
                             seed=1000 + k, eps=eps)[0] for k in range(40)])
    print(eps, est.mean(),
          np.sqrt((est.mean() - truth)**2 + est.std(ddof=1)**2))


V     = np.full(n_paths, heston_call(S0, K, T, r, q, v0, kappa, theta, sigma, rho))
delta = bs_delta(S[0], K, T, r, q, iv)
cash  = V - delta*S[0]

port = np.zeros((n_rebal + 1, n_paths))

for i in range(1, n_rebal + 1):
    tau = T - i*dt

    cash = cash*np.exp(r*dt)
    cash = cash + delta*S[i]*(np.exp(q*dt) - 1)

    if i < n_rebal:
        V = np.array([heston_call(S[i][p], K, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
    else:
        V = np.maximum(S[i] - K, 0)

    port[i] = -V + delta*S[i] + cash

    if i < n_rebal:
        delta_new = bs_delta(S[i], K, tau, r, q, iv)
        cash      = cash - (delta_new - delta)*S[i]
        delta     = delta_new

pnl_gbm = port[n_rebal]
print(pnl_gbm.mean(), np.sqrt((pnl_gbm**2).mean()))

days = np.linspace(365*T, 0, n_rebal + 1)

plt.plot(days, port[:, ::12], lw=0.4, color='0.7')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()

plt.plot(days, np.quantile(port, 0.95, axis=1), ls='--', color='navy')
plt.plot(days, np.quantile(port, 0.50, axis=1),          color='navy')
plt.plot(days, np.quantile(port, 0.05, axis=1), ls=':',  color='navy')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()


V = np.full(n_paths, heston_call(S0, K, T, r, q, v0, kappa, theta, sigma, rho))
delta = delta_mc(S[0], v[0], T, K, seed=70000)
cash  = V - delta*S[0]

port = np.zeros((n_rebal + 1, n_paths))

for i in range(1, n_rebal + 1):
    tau = T - i*dt

    cash = cash*np.exp(r*dt)
    cash = cash + delta*S[i]*(np.exp(q*dt) - 1)

    if i < n_rebal:
        V = np.array([heston_call(S[i][p], K, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
    else:
        V = np.maximum(S[i] - K, 0)

    port[i] = -V + delta*S[i] + cash

    if i < n_rebal:
        delta_new = delta_mc(S[i], v[i], tau, K, seed=70000 + i)
        cash      = cash - (delta_new - delta)*S[i]
        delta     = delta_new

pnl_fd = port[n_rebal]
print(pnl_fd.mean(), np.sqrt((pnl_fd**2).mean()))

days = np.linspace(365*T, 0, n_rebal + 1)

plt.plot(days, port[:, ::12], lw=0.4, color='0.7')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()

plt.plot(days, np.quantile(port, 0.95, axis=1), ls='--', color='darkred')
plt.plot(days, np.quantile(port, 0.50, axis=1),          color='darkred')
plt.plot(days, np.quantile(port, 0.05, axis=1), ls=':',  color='darkred')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()


print(mae_es(pnl_gbm))
print(mae_es(pnl_fd))

rmse = lambda x: np.sqrt((x**2).mean())
mae  = lambda x: np.abs(x).mean()
es95 = lambda x: x[x <= np.quantile(x, 0.05)].mean()

print(paired_bootstrap(pnl_gbm, pnl_fd, rmse))
print(paired_bootstrap(pnl_gbm, pnl_fd, mae))
print(paired_bootstrap(pnl_gbm, pnl_fd, es95))
print((np.abs(pnl_fd) > np.abs(pnl_gbm)).sum())


V = np.full(n_paths, heston_call(S0, K, T, r, q, v0,
                                     kappa, theta, sigma, rho))
delta = np.full(n_paths, heston_delta(S0, K, T, r, q, v0,
                                      kappa, theta, sigma, rho))
cash = V - delta*S[0]

port = np.zeros((n_rebal + 1, n_paths))

for i in range(1, n_rebal + 1):
    tau = T - i*dt

    cash = cash*np.exp(r*dt)
    cash = cash + delta*S[i]*(np.exp(q*dt) - 1)

    if i < n_rebal:
        V = np.array([heston_call(S[i][p], K, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
    else:
        V = np.maximum(S[i] - K, 0)

    port[i] = -V + delta*S[i] + cash

    if i < n_rebal:
        delta_new = np.array([heston_delta(S[i][p], K, tau, r, q, v[i][p],
                                           kappa, theta, sigma, rho)
                              for p in range(n_paths)])
        cash      = cash - (delta_new - delta)*S[i]
        delta     = delta_new

pnl = port[n_rebal]
print(pnl.mean(), np.sqrt((pnl**2).mean()))


V = np.full(n_paths, heston_call(S0, K_V, T, r, q, v0, kappa, theta, sigma, rho))
C = np.full(n_paths, heston_call(S0, K_C, T, r, q, v0, kappa, theta, sigma, rho))

h    = bs_vega(S[0], K_V, T, r, q, iv_V) / bs_vega(S[0], K_C, T, r, q, iv_C)
n_S  = bs_delta(S[0], K_V, T, r, q, iv_V) - h*bs_delta(S[0], K_C, T, r, q, iv_C)
cash = V - n_S*S[0] - h*C

floor = 0.01 * bs_vega(S[0], K_C, T, r, q, iv_C)
port  = np.zeros((n_rebal + 1, n_paths))

for i in range(1, n_rebal + 1):
    tau = T - i*dt

    cash = cash*np.exp(r*dt)
    cash = cash + n_S*S[i]*(np.exp(q*dt) - 1)

    if i < n_rebal:
        V = np.array([heston_call(S[i][p], K_V, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
        C = np.array([heston_call(S[i][p], K_C, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
    else:
        V = np.maximum(S[i] - K_V, 0)
        C = np.maximum(S[i] - K_C, 0)

    port[i] = -V + n_S*S[i] + h*C + cash

    if i < n_rebal:
        vega_V = bs_vega(S[i], K_V, tau, r, q, iv_V)
        vega_C = bs_vega(S[i], K_C, tau, r, q, iv_C)
        d_V    = bs_delta(S[i], K_V, tau, r, q, iv_V)
        d_C    = bs_delta(S[i], K_C, tau, r, q, iv_C)

        live   = vega_C >= floor
        h_new  = np.where(live, vega_V/np.where(live, vega_C, 1.0), 0.0)
        n_new  = d_V - h_new*d_C

        cash   = cash - (h_new - h)*C - (n_new - n_S)*S[i]
        h, n_S = h_new, n_new

pnl_dv_gbm = port[n_rebal]
print(pnl_dv_gbm.mean(), np.sqrt((pnl_dv_gbm**2).mean()), mae_es(pnl_dv_gbm))

days = np.linspace(365*T, 0, n_rebal + 1)

plt.plot(days, port[:, ::12], lw=0.4, color='0.7')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()

plt.plot(days, np.quantile(port, 0.95, axis=1), ls='--', color='navy')
plt.plot(days, np.quantile(port, 0.50, axis=1),          color='navy')
plt.plot(days, np.quantile(port, 0.05, axis=1), ls=':',  color='navy')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()


V = np.full(n_paths, heston_call(S0, K_V, T, r, q, v0,
                                 kappa, theta, sigma, rho))
C = np.full(n_paths, heston_call(S0, K_C, T, r, q, v0,
                                 kappa, theta, sigma, rho))

d_V, vega_V = mc_greeks(S[0], v[0], T, K_V, seed=50000)
d_C, vega_C = mc_greeks(S[0], v[0], T, K_C, seed=50000)

h    = vega_V / vega_C
n_S  = d_V - h*d_C
cash = V - n_S*S[0] - h*C

floor = 0.01 * vega_C
port  = np.zeros((n_rebal + 1, n_paths))

for i in range(1, n_rebal + 1):
    tau = T - i*dt

    cash = cash*np.exp(r*dt)
    cash = cash + n_S*S[i]*(np.exp(q*dt) - 1)

    if i < n_rebal:
        V = np.array([heston_call(S[i][p], K_V, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
        C = np.array([heston_call(S[i][p], K_C, tau, r, q, v[i][p],
                                  kappa, theta, sigma, rho)
                      for p in range(n_paths)])
    else:
        V = np.maximum(S[i] - K_V, 0)
        C = np.maximum(S[i] - K_C, 0)

    port[i] = -V + n_S*S[i] + h*C + cash

    if i < n_rebal:
        d_V, vega_V = mc_greeks(S[i], v[i], tau, K_V, seed=50000 + i)
        d_C, vega_C = mc_greeks(S[i], v[i], tau, K_C, seed=50000 + i)

        live   = vega_C >= floor
        h_new  = np.where(live, vega_V/np.where(live, vega_C, 1.0), 0.0)
        n_new  = d_V - h_new*d_C

        cash   = cash - (h_new - h)*C - (n_new - n_S)*S[i]
        h, n_S = h_new, n_new

pnl_dv_fd = port[n_rebal]
print(pnl_dv_fd.mean(), np.sqrt((pnl_dv_fd**2).mean()), mae_es(pnl_dv_fd))

days = np.linspace(365*T, 0, n_rebal + 1)

plt.plot(days, port[:, ::12], lw=0.4, color='0.7')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()

plt.plot(days, np.quantile(port, 0.95, axis=1), ls='--', color='darkred')
plt.plot(days, np.quantile(port, 0.50, axis=1),          color='darkred')
plt.plot(days, np.quantile(port, 0.05, axis=1), ls=':',  color='darkred')
plt.axhline(0, color='black', lw=1)
plt.gca().invert_xaxis()
plt.xlabel('Days to expiry'); plt.ylabel('Hedging error')
plt.show()


print(paired_bootstrap(pnl_dv_gbm, pnl_dv_fd, rmse))
print(paired_bootstrap(pnl_dv_gbm, pnl_dv_fd, mae))
print(paired_bootstrap(pnl_dv_gbm, pnl_dv_fd, es95))

