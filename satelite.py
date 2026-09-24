import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import newton

# --- 物理定数 ---
G = 6.6743e-11             # 万有引力定数 (m^3/kg/s^2)
sigma_sb = 5.67e-8         # ステファン・ボルツマン定数 (W/m^2/K^4)
r_gg = 1.7e-7              # 粒子とガスの面密度比
k_B = 1.380649e-23         # ボルツマン定数 (J/K)
m_g = 3.9e-27              # ガスの平均分子質量 (kg)
alpha = 1.0e-4              # 乱流の強さ
M_J = 1.90e27              # 木星質量（kg）
t_gap = 1.0e6              # ギャップ形成時の時間（yr）
R_J = 7.15e7               # 木星半径（m）
x = 0.0026                 # ダストとガスの降着比
C_D = 0.5                  # 抵抗係数 
b_t1 = -1.0                # Type I マイグレーションの係数（内側への移動が負）
    
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
    term_p_q2 = (abs(10.0 / (18.0 - 13.0 * q)))**(2/5)
    
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
# 1. シミュレーションの初期設定
# ==========================================
t_year = 2.0e6             # 初期時間 (年)
dt_year = 500.0             # 1タイムステップ = 1000年
t_max_year = 3.0e7      # シミュレーション期間 = 3000万年

# 衛星の初期状態
R_s = 3.35e5               # 初期半径 (m) 
rho_s = 1900.0            # 密度
C_p = 1400.0              # 比熱
r_orb = 50.0 * R_J        # 最初の衛星軌道半径 (m) 

# 記録用のリスト
history_t = []
history_Rs = []
history_Ts = []

# ==========================================
# 2. 時間発展のループ（メインの計算部分）
# ==========================================
print("シミュレーションを開始します...")

while t_year <= t_max_year:
    # --- 惑星と円盤のパラメータ更新 ---
    M_cp_MJ = calculate_M_cp(t_year, t_gap)
    M_cp_kg = M_cp_MJ * M_J
    
    M_dot_g_MJ_yr = calculate_mdot_g(t_year, t_gap)
    M_dot_g_kg_s = M_dot_g_MJ_yr * M_J / (365.25 * 24 * 3600) # SI単位に変換
    M_dot_p_kg_s = x * M_dot_g_kg_s
    
    # --- 円盤構造の計算 ---
    T_d, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_orb, alpha, r_gg)
    
    # ニュートン法が失敗した場合のフェイルセーフ
    if np.isnan(T_d):
        print(f"Time {t_year}: 温度計算が収束しませんでした。計算を停止します。")
        break
        
    eta, H_g = calculate_eta(M_cp_kg, M_dot_g_kg_s, r_orb, alpha, r_gg)
    
    v_K = np.sqrt(G * M_cp_kg / r_orb)
    
    # 衛星質量の概算 (体積 * 密度)
    M_s = (4.0 / 3.0) * np.pi * R_s**3 * rho_s
    mu_s = calculate_mu_s(M_s, M_cp_kg)
    p, q = calculate_p_q(M_cp_kg, M_dot_g_kg_s, r_orb, alpha, r_gg) 
    st_p = calculate_st_p(p, q, M_dot_p_kg_s, M_dot_g_kg_s, alpha, T_d, M_cp_MJ, r_orb/R_J)

    H_p = H_g * (1.0 + (st_p / alpha) * (1.0 + 2.0 * st_p) / (1.0 + st_p))**(-0.5)
    h_p = H_p / r_orb

    delta_v = calcuu_i = calculate_delta_v(mu_s, eta, st_p, v_K)
    
    Omega_K_current = v_K / r_orb
    u_i = calculate_u_i(G, M_s, R_s, delta_v, st_p, Omega_K_current)

    # --- 質量降着率と成長率の計算 ---
    P_eff = calculate_P_eff(mu_s, delta_v, v_K, st_p, eta, h_p)
    M_dot_s = M_dot_p_kg_s * P_eff
    
    dRs_dt = M_dot_s / (4 * np.pi * rho_s * R_s**2)

    # マイグレーション計算に必要な物理量（音速、ガス空間密度）の導出
    c_s = np.sqrt((k_B * T_d) / m_g)
    rho_g = Sigma_g / (np.sqrt(2.0 * np.pi) * H_g)
        
    # 衛星のストークス数と各移動速度 (m/s) の計算
    St_s = calculate_satellite_stokes_number(C_D, rho_s, R_s, rho_g, eta, v_K, Omega_K_current)
    v_ad = calculate_v_ad(St_s, eta, v_K)
    v_t1 = calculate_v_t1(b_t1, M_s, M_cp_kg, Sigma_g, r_orb, v_K, c_s)
    
    # 合計の軌道移動速度 (m/s)
    v_mig = v_ad + v_t1
    # ==========================================

    # --- 表面温度の計算 ---
    Ts_strict, Ts_guess = calculate_Ts_strict(rho_s, C_p, dRs_dt, M_dot_s, u_i, R_s, T_d)
    
    # --- データの記録 ---
    history_t.append(t_year)
    history_Rs.append(R_s / 1000.0)  # m から km に変換して記録
    history_Ts.append(Ts_strict)
    
    # --- 状態の更新（オイラー法）---
    r_orb += v_mig * (dt_year * 365.25 * 24 * 3600)
    R_s += dRs_dt * (dt_year * 365.25 * 24 * 3600)  # dRs_dt は m/s なので秒を掛ける
    t_year += dt_year
    # 一定の軌道でマイグレーションを停止
    r_stop = 23.6 * R_J  # 
    if r_orb < r_stop:
        r_orb = r_stop  # 軌道半径をストップ位置に固定する

# ==========================================
# 3. 計算終了後、結果を出力・グラフ化
# ==========================================
if len(history_Rs) > 0:
    print("計算完了！")
    print(f"3000万年後の衛星半径: {history_Rs[-1]:.2f} km")
    print(f"3000万年後の表面温度: {history_Ts[-1]:.2f} K")

    # グラフ描画
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Radius (km)', color=color)
    ax1.plot(history_t, history_Rs, color=color, label='Radius')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Surface Temp (K)', color=color)  
    ax2.plot(history_t, history_Ts, color=color, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Satellite Growth over 3.0e7 Years')
    plt.show()
