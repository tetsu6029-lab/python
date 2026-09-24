import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import newton

plt.rcParams['font.family'] = 'IPAexGothic'

# --- 物理定数 ---
G = 6.6743e-11             # 万有引力定数 (m^3/kg/s^2)
sigma_sb = 5.67e-8         # ステファン・ボルツマン定数 (W/m^2/K^4)
r_gg = 1.7e-7              # 粒子とガスの面密度比
k_B = 1.380649e-23         # ボルツマン定数 (J/K)
m_g = 3.9e-27              # ガスの平均分子質量 (kg)
alpha = 1.0e-4              # 乱流の強さ
M_J = 1.90e27              # 木星質量（kg）
t_gap = 1.0e6              # ギャップ形成時の時間（yr）※単位調整
R_J = 7.15e7               # 木星半径（m）※kmからmに統一
x = 0.0026                 # ダストとガスの降着比
C_D = 0.5                  # 抵抗係数 
b_t1 = -0.8                # Type I マイグレーションの係数（内側への移動を負とする）
C_p = 1400.0               # 比熱

# ==========================================
# 事前準備：物理量を計算する関数群
# ==========================================

# 時間t[yr]における中心惑星質量 M_cp [M_J] を計算する関数
def calculate_M_cp(t, t_gap):
    M_cp_gap = 0.4                  # t_gap における初期質量 [M_J]
    M_dot_g_gap = 0.2e-6            # ギャップ形成時のガス降着率 [M_J / yr]
    t_dep = 3.0e6                   # ガス枯渇のタイムスケール [yr]

    if t <= t_gap:
        return M_cp_gap
    else:
        M_cp = M_cp_gap + M_dot_g_gap * t_dep * (1 - math.exp(-(t - t_gap) / t_dep))
        return M_cp

# 衛星質量 M_s と 中心惑星質量 M_cp から質量比 mu_s を計算する関数
def calculate_mu_s(M_s, M_cp): 
    return M_s / M_cp

# ガスのフラックス (M_dot_g) を計算する関数
def calculate_mdot_g(t, t_gap):
    M_dot_g_gap = 0.2e-6
    t_dep = 3.0e6
    if t > t_gap:
        M_dot_g = M_dot_g_gap * math.exp(-(t - t_gap) / t_dep)
    else:
        M_dot_g = M_dot_g_gap
    return M_dot_g


# 円盤ガスの中面温度（T_d）、面密度を計算する関数
def calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r, alpha, r_gg):
    Omega_K = np.sqrt(G * M_cp_kg / r**3)
    T_d0 = ((3.0 * G * M_cp_kg * M_dot_g_kg_s) / (8.0 * np.pi * sigma_sb * r**3))**0.25
    
    def balance_equation(T_d):
        if T_d <= 0:
            return -1e10
        c_s = np.sqrt((k_B * T_d) / m_g)
        Sigma_g = (M_dot_g_kg_s * Omega_K) / (3.0 * np.pi * alpha * c_s**2)
        if T_d >= 160.0:
            kappa = 450.0 * r_gg
        else:
            kappa = 450.0 * (T_d / 160.0)**2 * r_gg
        tau = kappa * Sigma_g
        tau = max(tau, 1e-10)
        g = ((3.0 / 8.0) * tau + 1.0 / (4.8 * tau))**0.25
        return T_d - (T_d0 * g)

    try:
        T_d_solution = newton(balance_equation, x0=T_d0)
    except RuntimeError:
        print("警告: ニュートン法が収束しませんでした。")
        return np.nan, np.nan
        
    c_s_final = np.sqrt((k_B * T_d_solution) / m_g)
    Sigma_g_final = (M_dot_g_kg_s * Omega_K) / (3.0 * np.pi * alpha * c_s_final**2)
    return T_d_solution, Sigma_g_final

# 中心差分法を用いて、その半径 r における局所的なべき指数 p と q を計算する関数
def calculate_p_q(M_cp_kg, M_dot_g_kg_s, r, alpha, r_gg):
      
    # 微小な半径のズレ 
    dr = r * 1e-3
    r_minus = r - dr
    r_plus  = r + dr
    
    # 少し内側(r_minus)と少し外側(r_plus)の温度と面密度を計算
    T_d_minus, Sigma_g_minus = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_minus, alpha, r_gg)
    T_d_plus, Sigma_g_plus   = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_plus, alpha, r_gg)
    
    # --- p の計算 (Sigma_g の傾き) ---
    d_ln_Sigma_g = np.log(Sigma_g_plus) - np.log(Sigma_g_minus)
    d_ln_r = np.log(r_plus) - np.log(r_minus)
    p = -(d_ln_Sigma_g / d_ln_r)
    
    # --- q の計算 (T_d の傾き) ---
    d_ln_T_d = np.log(T_d_plus) - np.log(T_d_minus)
    q = -(d_ln_T_d / d_ln_r)

    return p, q

# ぺブルのストークス数 (St_p) を計算する関数
def calculate_st_p(p, q, M_dot_p, M_dot_g, alpha, T_d, M_cp_MJ, r_RJ):
    coef = 0.23
    term_p_q1 = (2.0 / (3.0 + 2.0 * p + q))**(4/5)
    term_p_q2 = (abs(10.0 / (18.0 - 39.0 * q)))**(2/5)
    # 0割り防止
    if M_dot_g <= 0: return 1e-10
    
    M_dot_ratio = M_dot_p / M_dot_g
    term_mdot = (M_dot_ratio / 0.003)**(2/5)
    term_alpha = (alpha / 1e-4)**(1/5)
    term_td = (T_d / 160.0)**(-2/5)
    term_mcp = (M_cp_MJ / 1.0)**(2/5)
    term_r = (r_RJ / 10.0)**(-2/5)
    
    st_p = coef * term_p_q1 * term_p_q2 * term_mdot * term_alpha * term_td * term_mcp * term_r
    return st_p

# 圧力勾配によるずれ（eta）を計算する関数
def calculate_eta(M_cp_kg, M_dot_g_kg_s, r, alpha, r_gg):
    def get_pressure(radius):
        T_d, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, radius, alpha, r_gg)
        Omega_K = np.sqrt(G * M_cp_kg / radius**3)
        c_s = np.sqrt((k_B * T_d) / m_g)
        H_g = c_s / Omega_K
        rho_g = Sigma_g / (np.sqrt(2.0 * np.pi) * H_g)
        P = rho_g * c_s**2
        return P, H_g
        
    dr = r * 1e-3 
    r_minus = r - dr
    r_plus  = r + dr
    
    P_minus, _ = get_pressure(r_minus)
    P_plus,  _ = get_pressure(r_plus)
    P_center, H_g_center = get_pressure(r)
    
    d_ln_P = np.log(P_plus) - np.log(P_minus)
    d_ln_r = np.log(r_plus) - np.log(r_minus)
    dlnP_dlnr = d_ln_P / d_ln_r
    
    eta = -0.5 * (H_g_center / r)**2 * dlnP_dlnr
    return eta, H_g_center

# 衛星のストークス数 (St_s) を計算する関数

def calculate_satellite_stokes_number(C_D, rho_s, R_s, rho_g, eta, v_K, Omega_K):
    
    # 8 / 3C_D の部分
    coef = 8.0 / (3.0 * C_D)

    # 密度や半径、速度の項
    fraction = (rho_s * R_s) / (rho_g * eta * v_K)

    # 全てを掛け合わせる
    St_s = coef * fraction * Omega_K
    return St_s


# 空気力学的抵抗による移動速度 (v_ad) を計算する関数
def calculate_v_ad(St_s, eta, v_K):

    # -2 * (St_s / (St_s^2 + 1)) の部分
    fraction = St_s / (St_s**2 + 1.0)

    v_ad = -2.0 * fraction * eta * v_K
    return v_ad


# Type I マイグレーション速度 (v_t1) を計算する関数

def calculate_v_t1(b_t1, M_s, M_cp, Sigma_g, r, v_K, c_s):

    # 3つの括弧の項をそれぞれ計算
    term1 = M_s / M_cp
    term2 = (Sigma_g * r**2) / M_cp
    term3 = (v_K / c_s)**2

    # 全てを掛け合わせる
    v_t1 = b_t1 * term1 * term2 * term3 * v_K
    return v_t1


# マイグレーションのタイムスケール (t_mig) を計算する関数: t_mig = r / |v_ad + v_t1|

def calculate_t_mig(r, v_ad, v_t1):

    # 移動速度の絶対値を取る（ゼロ割りを防ぐための微小値も念のため追加）
    total_v = abs(v_ad + v_t1) + 1e-20

    t_mig = r / total_v
    return t_mig

# 衛星とぺブルの相対速度delta_vを計算する関数
def calculate_delta_v(mu_s, eta, st_p, v_K):
    term1 = 0.52 * (mu_s * st_p)**(1/3)
    inner_fraction = mu_s / (eta**3 / st_p)
    term2 = eta * (1 + 5.7 * inner_fraction)**-1
    delta_v_over_v_K = term1 + term2
    delta_v = delta_v_over_v_K * v_K
    return delta_v

# 衛星とペブルの衝突速度 (u_i) を計算する関数

def calculate_u_i(G, M_s, R_s, delta_v, st_p, Omega_K):

    # 衛星表面の重力加速度 (g_s)
    g_s = (G * M_s) / (R_s**2)
    
    # 脱出速度 (v_esc)
    v_esc = np.sqrt(2.0 * G * M_s / R_s)
    
    # ペブルの停止時間 (t_stop)
    t_stop = st_p / Omega_K
    
    # 沈降速度 (v_set)
    v_set = g_s * t_stop
    
    # 衝突速度 (u_i) : min(√(v_esc^2 + Δv^2), v_set)
    u_i = min(np.sqrt(v_esc**2 + delta_v**2), v_set)
    
    return u_i

# P_effを計算する関数
def calculate_P_eff(mu_s, delta_v, v_K, st_p, eta, h_p):
    term1_inner = 0.32 * math.sqrt(abs((mu_s * delta_v / v_K) / (st_p * eta**2)))
    term2_inner = 0.39 * (mu_s / (eta * h_p))
    P_eff = (term1_inner**-2 + term2_inner**-2)**-0.5
    return P_eff

# ニュートン法による厳密解
def calculate_Ts_strict(rho_s, C_p, dRs_dt, M_dot_s, u_i, R_s, T_d):
    heating_flux = (M_dot_s * u_i**2) / (8 * np.pi * R_s**2)
    T_s_guess = ((heating_flux / sigma_sb) + T_d**4)**0.25
    
    heat_capacity_term = rho_s * C_p * dRs_dt
    constant_cooling_term = heating_flux + sigma_sb * T_d**4 + heat_capacity_term * T_d
    
    def energy_balance(T_s):
        return sigma_sb * T_s**4 + heat_capacity_term * T_s - constant_cooling_term
    
    T_s_strict = newton(energy_balance, x0=T_s_guess)
    return T_s_strict, T_s_guess

# ==========================================
# 1. 複数衛星を管理するクラスの定義
# ==========================================
class Satellite:
    def __init__(self, name, t_cap, r_ini):
        self.name = name
        self.t_cap = t_cap        # 捕獲される（計算が始まる）時間
        self.r_orb = r_ini        # 現在の軌道半径
        self.R_s = 3.35e5         # 初期半径 (m)
        self.rho_s = 1900.0       # 密度
        self.M_s = (4.0/3.0) * np.pi * self.R_s**3 * self.rho_s
        
        # 記録用リスト
        self.history_t, self.history_Rs, self.history_r, self.history_Ts = [], [], [], []

# シミュレーションの初期設定
t_year = 1.0e6             # 最初の衛星（イオ）が捕獲される時間からスタート
dt_year = 1000.0            # タイムステップ
t_max_year = 3.0e7         # 3000万年

# 4つの衛星（シード）を定義（Shibaike et al. 2019 に基づく時間差）
satellites = [
    Satellite("Io",       1.00e6, 50.0 * R_J),
    Satellite("Europa",   1.25e6, 50.0 * R_J),
    Satellite("Ganymede", 1.50e6, 50.0 * R_J),
    Satellite("Callisto", 2.00e6, 50.0 * R_J)
]

# ==========================================
# 2. 時間発展のループ（メインの計算部分）
# ==========================================
print("シミュレーションを開始します...")

while t_year <= t_max_year:
    M_cp_MJ = calculate_M_cp(t_year, t_gap)
    M_cp_kg = M_cp_MJ * M_J
    M_dot_g_kg_s = calculate_mdot_g(t_year, t_gap) * M_J / (365.25 * 24 * 3600)
    M_dot_p_base = x * M_dot_g_kg_s
    
    # すでに捕獲済みの衛星を抽出し、軌道が「外側」の順に並び替え
    active_sats = [sat for sat in satellites if t_year >= sat.t_cap]
    active_sats.sort(key=lambda s: s.r_orb, reverse=True)
    
    current_M_dot_p = M_dot_p_base
    
    # --- ① 物理計算（外側から順番にペブルを横取りしていく） ---
    for sat in active_sats:
        T_d, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg)
        if np.isnan(T_d): continue
            
        # スノーラインの判定（160K以上なら氷が蒸発してペブル質量フラックスが半減）
        local_M_dot_p = current_M_dot_p * 0.5 if T_d >= 160.0 else current_M_dot_p
        
        p, q = calculate_p_q(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg)
        st_p = calculate_st_p(p, q, local_M_dot_p, M_dot_g_kg_s, alpha, T_d, M_cp_MJ, sat.r_orb/R_J)
        
        eta, H_g = calculate_eta(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg)
        v_K = np.sqrt(G * M_cp_kg / sat.r_orb)
        Omega_K = v_K / sat.r_orb
        
        H_p = H_g * (1.0 + (st_p / alpha) * (1.0 + 2.0 * st_p) / (1.0 + st_p))**(-0.5)
        h_p = H_p / sat.r_orb
        
        mu_s = calculate_mu_s(sat.M_s, M_cp_kg)
        delta_v = calculate_delta_v(mu_s, eta, st_p, v_K)
        P_eff = calculate_P_eff(mu_s, delta_v, v_K, st_p, eta, h_p)
        
        # 【重要】自分が捕獲した分、さらに内側に流れるペブルが減る（フィルタリング）
        current_M_dot_p = current_M_dot_p * (1.0 - P_eff)
        
        M_dot_s = local_M_dot_p * P_eff
        dRs_dt = M_dot_s / (4 * np.pi * sat.rho_s * sat.R_s**2)
        
        c_s = np.sqrt((k_B * T_d) / m_g)
        rho_g = Sigma_g / (np.sqrt(2.0 * np.pi) * H_g)
        St_s = calculate_satellite_stokes_number(C_D, sat.rho_s, sat.R_s, rho_g, eta, v_K, Omega_K)
        v_mig = calculate_v_ad(St_s, eta, v_K) + calculate_v_t1(b_t1, sat.M_s, M_cp_kg, Sigma_g, sat.r_orb, v_K, c_s)
        
        u_i = calculate_u_i(G, sat.M_s, sat.R_s, delta_v, st_p, Omega_K)
        Ts_strict, _ = calculate_Ts_strict(sat.rho_s, C_p, dRs_dt, M_dot_s, u_i, sat.R_s, T_d)
        
        # 次のステップの値を一時保存
        sat.next_r_orb = sat.r_orb + v_mig * (dt_year * 365.25 * 24 * 3600)
        sat.next_R_s = sat.R_s + dRs_dt * (dt_year * 365.25 * 24 * 3600)
        sat.next_Ts = Ts_strict
        sat.next_M_s = (4.0 / 3.0) * np.pi * sat.next_R_s**3 * sat.rho_s

    # --- ② 軌道共鳴チェック（内側の衛星から外側へ順番に判定） ---
    active_sats.sort(key=lambda s: s.r_orb)
    
    for i, sat in enumerate(active_sats):
        # イオはキャビティ（空洞の端）で止まる
        if sat.next_r_orb < 5.89 * R_J:
            sat.next_r_orb = 5.89 * R_J
            
        # 2:1軌道共鳴の判定（すぐ内側の衛星の 1.5874倍 の位置で止まる）
        if i > 0:
            resonance_pos = active_sats[i-1].next_r_orb * (2.0**(2/3))
            if sat.next_r_orb < resonance_pos:
                sat.next_r_orb = resonance_pos
                
        # 状態を確定し、リストに記録
        sat.r_orb, sat.R_s, sat.M_s = sat.next_r_orb, sat.next_R_s, sat.next_M_s
        sat.history_t.append(t_year)
        sat.history_Rs.append(sat.R_s / 1000.0)
        sat.history_r.append(sat.r_orb / R_J)
        sat.history_Ts.append(sat.next_Ts)

    t_year += dt_year

# ==========================================
# 3. 計算終了後、結果を出力・グラフ化
# ==========================================
colors = ['tab:orange', 'tab:cyan', 'tab:blue', 'tab:gray']

# --- 1枚目: 軌道進化 (Orbital Evolution) ---
plt.figure(figsize=(8, 5))  # 個別のグラフのサイズを指定
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_r, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('Orbital Radius ($R_J$)')
plt.xlabel('Time (Years)')
plt.title('軌道進化 (Resonance)')
plt.xscale('log')           # 独立したグラフになるのでそれぞれにX軸設定が必要
plt.legend()
plt.tight_layout()
plt.savefig("orbital_evolution.png")
plt.close()                 # メモリを解放して次のキャンバスへ

# --- 2枚目: 成長履歴 (Growth History) ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_Rs, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('Satellite Radius (km)')
plt.xlabel('Time (Years)')
plt.title('半径進化')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("growth_history.png")
plt.close()

# --- 3枚目: 表面温度 (Thermal Evolution) ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_Ts, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('Surface Temp (K)')
plt.xlabel('Time (Years)')
plt.title('表面温度')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("thermal_evolution.png")
plt.close()
