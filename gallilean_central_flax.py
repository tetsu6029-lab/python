import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate
from scipy.optimize import newton

plt.rcParams['font.family'] = 'IPAexGothic'

# --- 物理定数 ---
G = 6.6743e-8             # 万有引力定数 (cm^3/g/s^2)
sigma_sb = 5.67e-5         # ステファン・ボルツマン定数 (W/cm^2/K^4)
r_gg = 3.4e-7              # 粒子とガスの面密度比
k_B = 1.380649e-16         # ボルツマン定数 (erg/K)
m_g = 3.9e-24              # ガスの平均分子質量 (g)
alpha = 1.0e-4              # 乱流の強さ
M_J = 1.90e30              # 木星質量（g）
t_gap = 1.0e6              # ギャップ形成時の時間（yr）
R_J = 7.15e9               # 木星半径（cm）
x = 0.0026                 # ダストとガスの降着比
C_D = 0.5                  # 抵抗係数 
C_p = 1.4e7               # 比熱 (erg/g/K)
L_sun = 3.828e33          # 太陽光度 (erg/s) ※CGS単位系に合わせました
s_const = 2.0             # zs / Hg の比（円盤表面の高さを決める定数）
gamma = 1.4
au_cm = 1.495978707e13     # 1天文単位 (cm)
a_J = 5.2 * au_cm          # 太陽から木星までの距離 (cm)
L_sun_early = 0.70 * L_sun # 形成初期の太陽光度 (現在の75%)
T_bg = (L_sun_early / (16.0 * np.pi * a_J**2 * sigma_sb))**0.25 # 太陽輻射による木星軌道付近の平衡温度 
rho_ice = 1.4              # 氷の密度 (g/cm^3)
rho_rock = 3.0             # 岩石の密度 (g/cm^3)

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

# 木星の光度 L_cp [erg/s] を時間発展で計算する関数(Marley 2007)
def calculate_L_cp(t, t_gap):
    t_end = 3.0e7
    L_gap = 1.5e-5 * L_sun      # ギャップ形成時の光度
    L_end = 3.0e-7 * L_sun      # 計算終了時(t = 30Myr)の光度
    
    if t <= t_gap:
        return L_gap
    elif t >= t_gap:
        # 冷却モデルに合わせて対数スケール（指数関数的）で滑らかに減少させる
        log_L = math.log10(L_gap) + (t - t_gap) / (t_end - t_gap) * (math.log10(L_end) - math.log10(L_gap))
        return 10**log_L
    

# 円盤ガスの中面温度（T_d）、面密度を計算する関数
# q_guess: 温度勾配の推定値（放射優勢円盤の標準値 0.5 をデフォルトとする）
def calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r, alpha, r_gg, L_cp, q_guess=0.5):
    # ガス枯渇時の安全装置
    if M_dot_g_kg_s <= 1e-15:
        return 10.0, 1e-15

    Omega_K = np.sqrt(G * M_cp_kg / r**3)
    
    # 粘性加熱のエネルギー (Q / 2)
    Q_half = (3.0 * G * M_cp_kg * M_dot_g_kg_s) / (8.0 * np.pi * r**3)
    
    def balance_equation(T_d):
        if T_d <= 0:
            return -1e10
            
        c_s = np.sqrt((k_B * T_d) / m_g)
        H_g = c_s / Omega_K
        
        # 中心星からの輻射エネルギー (F_in)
        derivative_term = 0.5 * s_const * (H_g / r) * (1.0 - q_guess)
        F_in = 0.5 * (L_cp / (4.0 * np.pi * r**2)) * derivative_term

        #  太陽からの輻射エネルギーフラックス (F_sun)
        F_sun = sigma_sb * T_bg**4

        # 面密度とガス圧力の計算
        Sigma_g = (M_dot_g_kg_s * Omega_K) / (3.0 * np.pi * alpha * c_s**2)
        rho_g = Sigma_g / (np.sqrt(2.0 * np.pi) * H_g)
        P_gas = rho_g * c_s**2

        # --- 追加: クラウジウス・クラペイロン式による動的スノーライン計算 ---
        X_H2O = 1.7e-4  # 水のモル分率 (Oka et al. 2011により参考)
        P_H2O = max(X_H2O * P_gas, 1e-20)  # ゼロ割り防止
        T_snow = 6070.0 / (30.86 - np.log(P_H2O))

        # オパシティ
        if T_d < T_snow:
            kappa = 450.0 * (T_d / T_snow)**2 * r_gg  # 氷が存在する領域
        else:
            kappa = 450.0 * r_gg                      # 岩石のみの領域
            
        tau = kappa * Sigma_g
        tau = max(tau, 1e-10)
        
        # 【修正ポイント】粘性加熱と輻射加熱を分離する
        # 1. 粘性加熱（内部の摩擦熱）には光学厚みの効果(g_visc)がかかる
        g_visc_4 = (3.0 / 8.0) * tau + 1.0 / (4.8 * tau)
        T_visc_4 = (Q_half / sigma_sb) * g_visc_4
        
        # 2. 輻射加熱（外部からの光）は直接足し合わせる
        T_irr_4 = F_in / sigma_sb
        T_sun_4 = F_sun / sigma_sb

        # 3. 両方のエネルギーを足して実際の温度にする
        T_d_calc = (T_visc_4 + T_irr_4 + T_sun_4)**0.25
        
        return T_d - T_d_calc

    # ニュートン法の初期推測値
    T_guess = (Q_half / sigma_sb)**0.25
    if T_guess <= 0:
        T_guess = 10.0
    
    try:
        T_d_solution = newton(balance_equation, x0=T_guess)
    except RuntimeError:
        print(f"警告: ニュートン法が収束しませんでした (r={r/R_J:.2f} R_J)")
        return np.nan, np.nan
        
    c_s_final = np.sqrt((k_B * T_d_solution) / m_g)
    Sigma_g_final = (M_dot_g_kg_s * Omega_K) / (3.0 * np.pi * alpha * c_s_final**2)
    
    return T_d_solution, Sigma_g_final


# 中心差分法を用いて、反復計算で正確なべき指数 p と q を計算する関数
def calculate_p_q(M_cp_kg, M_dot_g_kg_s, r, alpha, r_gg, L_cp):
    global q_total_count, q_non_converge_count  # ★ 必ず def の直下に書く
    q_total_count += 1                          # ★ その次にカウントアップ
    
    dr = r * 1e-3
    r_minus = r - dr
    r_plus  = r + dr
    
    # 最初の推測値
    q_current = 0.5
    p_current = 1.0 # pの初期推測値（今回は計算ループに影響しませんが念のため）
    
    # 最大10回まで反復計算をして、本当の q を探る
    for _ in range(10):
        # 今の q_current を使って少し内側と外側の温度を計算
        T_d_minus, Sigma_g_minus = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_minus, alpha, r_gg, L_cp, q_guess=q_current)
        T_d_plus, Sigma_g_plus   = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_plus, alpha, r_gg, L_cp, q_guess=q_current)
        
        # 新しい p と q を計算
        d_ln_Sigma_g = np.log(Sigma_g_plus) - np.log(Sigma_g_minus)
        d_ln_r = np.log(r_plus) - np.log(r_minus)
        p_new = -(d_ln_Sigma_g / d_ln_r)
        
        d_ln_T_d = np.log(T_d_plus) - np.log(T_d_minus)
        q_new = -(d_ln_T_d / d_ln_r)
        
        # 新しい q と古い q の差が 0.001 以下（ほぼ一致）になったらループ終了
        if abs(q_new - q_current) < 1e-3:
            q_current = q_new
            p_current = p_new
            break
            
        # 一致しなかったら、新しい q を次の計算に回す
        q_current = q_new
        p_current = p_new

        # ★ 追加: forループがbreakされずに10回完走してしまった場合のみここを通る
    else:
        q_non_converge_count += 1

    return p_current, q_current

# --- 追加: 円盤ガス総質量を計算する関数 ---
def calculate_disk_mass(M_cp_kg, M_dot_g_kg_s, alpha, r_gg, L_cp):
    r_in = 1.0 * R_J       # 円盤の内縁（木星表面）
    r_out = 150.0 * R_J    # 円盤の外縁（木星半径の150倍を仮定）

    # 内側から外側まで、対数間隔で50個の計算ポイントを作る
    r_array = np.logspace(np.log10(r_in), np.log10(r_out), 50)
    sigma_array = []

    # 各ポイントでの面密度を計算
    for r_val in r_array:
        _, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_val, alpha, r_gg, L_cp, q_guess=0.5)
        sigma_array.append(Sigma_g)

    sigma_array = np.array(sigma_array)

    # ∫ 2π r Σ_g dr を台形積分で計算
    integrand = 2.0 * np.pi * r_array * sigma_array
    mass_g = np.trapezoid(integrand, r_array)

    # グラム(g)から木星質量(M_J)の単位に変換して返す
    return mass_g / M_J


# ぺブルのストークス数 (St_p) を計算する関数（破壊効果を追加）
def calculate_st_p(p, q, M_dot_p, M_dot_g, alpha, T_d, M_cp_kg, r, eta):
    # 0割り防止
    if M_dot_g <= 0: return 1e-10

    # 1. 動径移動による限界 (Drift-limited)
    M_cp_MJ = M_cp_kg / M_J
    r_RJ = r / R_J

    coef = 0.23
    term_p_q1 = (2.0 / (3.0 + 2.0 * p + q))**(4/5)
    term_p_q2 = (abs(10.0 / (18.0 - 13.0 * q)))**(2/5)

    M_dot_ratio = M_dot_p / M_dot_g
    term_mdot = (M_dot_ratio / 0.003)**(2/5)
    term_alpha = (alpha / 1e-4)**(1/5)
    term_td = (T_d / 160.0)**(-2/5)
    term_mcp = (M_cp_MJ / 1.0)**(2/5)
    term_r = (r_RJ / 10.0)**(-2/5)

    st_p_drift = coef * term_p_q1 * term_p_q2 * term_mdot * term_alpha * term_td * term_mcp * term_r

    # 2. 衝突破壊による限界 (Fragmentation-limited) - 論文の式(14)
    c_s = np.sqrt((k_B * T_d) / m_g)
    Omega_K = np.sqrt(G * M_cp_kg / r**3)
    eta_safe = max(abs(eta), 1e-15) # ゼロ割り回避

    # --- 動的スノーラインの計算 ---
    H_g_temp = c_s / Omega_K
    Sigma_g_temp = (M_dot_g / Omega_K) / (3.0 * np.pi * alpha * c_s**2)
    rho_g_temp = Sigma_g_temp / (np.sqrt(2.0 * np.pi) * H_g_temp)
    P_gas_temp = rho_g_temp * c_s**2
    
    X_H2O = 1.7e-4
    P_H2O = max(X_H2O * P_gas_temp, 1e-20)
    T_snow = 6070.0 / (30.86 - np.log(P_H2O))

    # スノーラインを境に限界衝突速度(v_cr)を変更
    if T_d >= T_snow:
        v_cr = 500.0   # 岩石ペブルの破壊速度 (cm/s)
    else:
        v_cr = 5000.0  # 氷ペブルの破壊速度 (cm/s)

    term_a = -3.0 * alpha * c_s**2
    term_b = np.sqrt(9.0 * alpha**2 * c_s**4 + 4.0 * eta_safe**2 * r**2 * Omega_K**2 * v_cr**2)
    term_c = 2.0 * eta_safe**2 * r**2 * Omega_K**2

    st_p_frag = (term_a + term_b) / term_c

    # ドリフトと破壊のうち、より小さい（厳しい）制限を採用する
    st_p = min(st_p_drift, st_p_frag)
    return st_p

# 圧力勾配によるずれ（eta）を計算する関数
def calculate_eta(M_cp_kg, M_dot_g_kg_s, r, alpha, r_gg, L_cp):
    def get_pressure(radius):
        T_d, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, radius, alpha, r_gg, L_cp)
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

# ==========================================
# Paardekooper et al. (2011) に由来する関数群
# ==========================================
def F_func(p):
    """
    Eq (23): 飽和(saturation)に関連する減少関数
    """
    return 1.0 / (1.0 + (p / 1.3)**2)

def G_func(p):
    """
    Eq (30): カットオフ(cutoff)に関連する関数
    """
    # 分岐のしきい値を計算
    threshold = math.sqrt(8.0 / (45.0 * math.pi))
    
    if p < threshold:
        return (16.0 / 25.0) * ((45.0 * math.pi / 8.0)**0.75) * (p**1.5)
    else:
        return 1.0 - (9.0 / 25.0) * ((8.0 / (45.0 * math.pi))**(4.0/3.0)) * (p**(-8.0/3.0))

def K_func(p):
    """
    Eq (31): カットオフ(cutoff)に関連する関数
    """
    # 分岐のしきい値を計算
    threshold = math.sqrt(28.0 / (45.0 * math.pi))
    
    if p < threshold:
        return (16.0 / 25.0) * ((45.0 * math.pi / 28.0)**0.75) * (p**1.5)
    else:
        return 1.0 - (9.0 / 25.0) * ((28.0 / (45.0 * math.pi))**(4.0/3.0)) * (p**(-8.0/3.0))

# マイグレーション係数b
def calculate_x_s(gamma, M_s, M_cp, h_gas):
    """
    Eq (16): 無次元化された馬蹄形領域の半幅 x_s 
    """
    return (1.1 / (gamma**0.25)) * np.sqrt((M_s / M_cp) / h_gas)

def calculate_P_nu(Omega_K, r, x_s, nu):
    """
    Eq (14): 粘性パラメータ P_ν を計算
    nu: 動粘性係数 (通常は α * c_s * H_g で計算)
    """
    # 0割り防止
    if nu <= 0: return 1e10
    return (2.0 / 3.0) * np.sqrt((Omega_K * r**2 * x_s**3) / (2.0 * np.pi * nu))

def calculate_P_chi(Omega_K, r, x_s, chi):
    """
    Eq (15): 熱パラメータ P_χ を計算
    chi: 熱拡散係数
    """
    # 0割り防止
    if chi <= 0: return 1e10
    return np.sqrt((Omega_K * r**2 * x_s**3) / (2.0 * np.pi * chi))

def calculate_beta(p, q, gamma, P_nu, P_chi):
    """
    Eqs (10)-(13): マイグレーション係数 β を計算
    """
    # 局所エントロピー勾配 (画像内の定義より)
    xi = q - (gamma - 1.0) * p
    
    # Eq (11): リンドブラッドトルク (Lindblad torque)
    beta_L = (2.0 / gamma) * (-2.5 - 1.7 * q + 0.1 * p)
    
    # Eq (12): コロテーショントルクのバロトロピック部分 (barotropic part)
    term1_baro = F_func(P_nu) * G_func(P_nu) * 1.1 * (1.5 - p)
    term2_baro = (1.0 - K_func(P_nu)) * 0.7 * (1.5 - p)
    beta_c_baro = (2.0 / gamma) * (term1_baro + term2_baro)
    
    # Eq (13): コロテーショントルクのエントロピー関連部分 (entropy-related part)
    # 平方根の中身が負にならないよう絶対値を取る(または max(0, x) を使う)フェイルセーフを追加
    G_prod = max(0.0, G_func(P_nu) * G_func(P_chi))
    K_prod = max(0.0, (1.0 - K_func(P_nu)) * (1.0 - K_func(P_chi)))
    
    term1_ent = F_func(P_nu) * F_func(P_chi) * np.sqrt(G_prod) * 7.9 * (xi / gamma)
    term2_ent = np.sqrt(K_prod) * (2.2 - 1.4 / gamma) * xi
    beta_c_ent = (2.0 / gamma) * (term1_ent + term2_ent)
    
    # Eq (10): 最終的な β の値
    b_t1 = beta_L + beta_c_baro + beta_c_ent
    
    return b_t1

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

# 衛星が保持する大気質量 (M_atm) を計算する関数
def calculate_atmosphere_mass(M_s, R_s, M_cp_kg, r_orb, c_s, rho_g):
    # ボンディ半径とヒル半径の計算
    R_B = (G * M_s) / (c_s**2)
    R_H = r_orb * (M_s / (3.0 * M_cp_kg))**(1/3)

    # 大気の上端は小さい方を採用
    R_out = min(R_B, R_H)

    # 衛星半径がR_outを上回っている場合は大気は存在できない
    if R_out <= R_s:
        return 0.0, R_out, R_B

    def density_profile(r):
        # 静水圧平衡による指数部の計算
        exponent = (G * M_s / c_s**2) * ((1.0 / r) - (1.0 / R_out))
        # オーバーフロー防止
        if exponent > 700:
            exponent = 700
        return rho_g * np.exp(exponent)

    def integrand(r):
        return 4.0 * np.pi * (r**2) * density_profile(r)

    # R_s から R_out まで数値積分
    mass_atm, _ = integrate.quad(integrand, R_s, R_out)

    return mass_atm, R_out, R_B

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
        self.R_s = 3.35e7         # 初期半径 (cm)
        self.rho_s = 1.9       # 密度 (cm/g)
        self.M_s = (4.0/3.0) * np.pi * self.R_s**3 * self.rho_s
        # ★ 追加: 蓄積された氷の総質量（初期の種は純粋な岩石と仮定）
        self.M_ice = self.M_s * 0.5
        self.M_rock = self.M_s * 0.5  
        
        # 記録用リスト
        self.history_t, self.history_Rs, self.history_r, self.history_Ts = [], [], [], []
        self.history_b_t1, self.history_T_d, self.history_P_eff, self.history_st_p = [], [], [], []
        self.history_u_i, self.history_q, self.history_P = [], [], []
        self.history_M_cp , self.history_v_ad ,self.history_t_ad = [], [], []
        self.history_R_B = []  # ★ 追加: ボンディ半径の履歴
        self.history_M_atm = []
        self.history_R_out = []
        self.history_R_B = []
        self.history_P_surf = []
        self.history_ice_frac = []

# シミュレーションの初期設定
t_year = 1.0e6             # イオが捕獲される時間からスタート
dt_year = 1000.0            # タイムステップ
t_max_year = 3.0e7         # 3000万年
# ★ 追加: qの計算の収束状況をカウントする変数
q_total_count = 0
q_non_converge_count = 0

# 追加: 固定軌道での温度を記録するための準備
target_fixed_r = 14.8 * R_J  # グラフ化したい軌道半径を指定
fixed_history_t = []
fixed_history_Td = []
fixed_history_T_visc = []
fixed_history_T_irr = []
fixed_history_T_sun = []
fixed_history_T_snow = []

history_disk_mass_t = []
history_disk_mass = []
history_M_cp = []
history_M_dot_g = []

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
    L_cp = calculate_L_cp(t_year, t_gap)
    current_disk_mass = calculate_disk_mass(M_cp_kg, M_dot_g_kg_s, alpha, r_gg, L_cp)
    history_disk_mass_t.append(t_year)
    history_disk_mass.append(current_disk_mass)
    history_M_cp.append(M_cp_MJ)
    history_M_dot_g.append(M_dot_g_kg_s)

    # 指定した固定軌道における各温度成分を計算して記録
    p_fix, q_fix = calculate_p_q(M_cp_kg, M_dot_g_kg_s, target_fixed_r, alpha, r_gg, L_cp)
    Td_fixed, Sigma_g_fixed = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, target_fixed_r, alpha, r_gg, L_cp, q_guess=q_fix)
    
    # ① 太陽からの熱 (T_sun)
    T_sun_val = T_bg
    
    # ② 木星からの熱 (T_irr)
    Omega_K_fixed = np.sqrt(G * M_cp_kg / target_fixed_r**3)
    c_s_fixed = np.sqrt((k_B * Td_fixed) / m_g)
    H_g_fixed = c_s_fixed / Omega_K_fixed
    derivative_term = 0.5 * s_const * (H_g_fixed / target_fixed_r) * (1.0 - q_fix)
    F_in_fixed = 0.5 * (L_cp / (4.0 * np.pi * target_fixed_r**2)) * derivative_term
    T_irr_val = (max(F_in_fixed, 0) / sigma_sb)**0.25
    
    # ③ 摩擦熱 (T_visc)
    T_visc_val = max(Td_fixed**4 - T_irr_val**4 - T_sun_val**4, 0.0)**0.25

    # ④ スノーラインの温度（T_snow）
    # まず空間密度(rho_g)とMKS単位系のガス圧(Pa)を計算
    rho_g_fixed = Sigma_g_fixed / (np.sqrt(2.0 * np.pi) * H_g_fixed)
    P_gas_cgs = rho_g_fixed * c_s_fixed**2
    
   
    
    # 動的スノーラインの計算 (Shibaike et al. 2019 に準拠したモル分率)
    X_H2O = 1.7e-4
    P_H2O_cgs = max(X_H2O * P_gas_cgs, 1e-20)
    T_snow_val = 6070.0 / (30.86 - np.log(P_H2O_cgs))

    fixed_history_t.append(t_year)
    fixed_history_Td.append(Td_fixed)
    fixed_history_T_visc.append(T_visc_val)
    fixed_history_T_irr.append(T_irr_val)
    fixed_history_T_sun.append(T_sun_val)
    fixed_history_T_snow.append(T_snow_val)

    # すでに捕獲済みの衛星を抽出し、軌道が「外側」の順に並び替え
    active_sats = [sat for sat in satellites if t_year >= sat.t_cap]
    active_sats.sort(key=lambda s: s.r_orb, reverse=True)
    
    current_M_dot_p = M_dot_p_base
    
    # --- ① 物理計算（外側から順番にペブルを横取りしていく） ---
    for sat in active_sats:

        p, q = calculate_p_q(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg, L_cp)
        T_d, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg, L_cp, q_guess=q)
        if np.isnan(T_d): continue
            
        # スノーラインの判定（T_snow以上なら氷が蒸発してペブル質量フラックスが半減）
        local_M_dot_p = current_M_dot_p * 0.5 if T_d >= T_snow_val else current_M_dot_p
       
        eta, H_g = calculate_eta(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg, L_cp)
        
        st_p = calculate_st_p(p, q, local_M_dot_p, M_dot_g_kg_s, alpha, T_d, M_cp_kg, sat.r_orb, eta)
        
        v_K = np.sqrt(G * M_cp_kg / sat.r_orb)
        Omega_K = v_K / sat.r_orb
        
        H_p = H_g * (1.0 + (st_p / alpha) * (1.0 + 2.0 * st_p) / (1.0 + st_p))**(-0.5)
        h_p = H_p / sat.r_orb

        mu_s = calculate_mu_s(sat.M_s, M_cp_kg)
        delta_v = calculate_delta_v(mu_s, eta, st_p, v_K)
        P_eff = calculate_P_eff(mu_s, delta_v, v_K, st_p, eta, h_p)
        
        # 【重要】自分が捕獲した分、さらに内側に流れるペブルが減る（フィルタリング）
        
        M_dot_s = local_M_dot_p * P_eff
        
        current_M_dot_p = current_M_dot_p - M_dot_s
        
        # --- 追加・修正: 氷と岩石の質量を個別に計算 ---
        # スノーラインより外側なら、降着するペブルの50%が氷だと仮定
        f_ice = 0.5 if T_d < T_snow_val else 0.0
        
        dt_sec = dt_year * 365.25 * 24 * 3600
        dM_s_total = M_dot_s * dt_sec
        
        sat.next_M_ice = sat.M_ice + dM_s_total * f_ice
        sat.next_M_rock = sat.M_rock + dM_s_total * (1.0 - f_ice)
        sat.next_M_s = sat.next_M_ice + sat.next_M_rock
        
        # 体積から新しい半径と平均密度を逆算
        V_next = (sat.next_M_ice / rho_ice) + (sat.next_M_rock / rho_rock)
        sat.next_R_s = ((3.0 / (4.0 * np.pi)) * V_next)**(1.0 / 3.0)
        sat.next_rho_s = sat.next_M_s / V_next
        
        # 降着するペブルの密度 (rho_in) を使って半径の成長速度を計算
        rho_in = 1.0 / (f_ice / rho_ice + (1.0 - f_ice) / rho_rock) if f_ice > 0 else rho_rock
        dRs_dt = M_dot_s / (4.0 * np.pi * rho_in * sat.R_s**2)
        
        c_s = np.sqrt((k_B * T_d) / m_g)
        rho_g = Sigma_g / (np.sqrt(2.0 * np.pi) * H_g)
        P =rho_g * c_s**2
        St_s = calculate_satellite_stokes_number(C_D, sat.rho_s, sat.R_s, rho_g, eta, v_K, Omega_K)

 # --- 追加: 大気質量と半径の計算 ---
        M_atm, R_out, R_B = calculate_atmosphere_mass(sat.M_s, sat.R_s, M_cp_kg, sat.r_orb, c_s, rho_g)
        St_s = calculate_satellite_stokes_number(C_D, sat.rho_s, sat.R_s, rho_g, eta, v_K, Omega_K)
        
        # ★ 追加: 表面大気圧の計算 (P_surf = rho(R_s) * c_s^2)
        if R_out > sat.R_s:
            # 表面での指数部を計算
            exponent = (G * sat.M_s / c_s**2) * ((1.0 / sat.R_s) - (1.0 / R_out))
            exponent = min(exponent, 700) # オーバーフロー防止
            rho_surf = rho_g * np.exp(exponent)
            P_surf_cgs = rho_surf * c_s**2  # 単位は dyn/cm^2
        else:
            P_surf_cgs = 0.0 # ボンディ半径が衛星半径より小さい場合は大気なし
        

        
        # 2. 画像の数式に基づく動粘性係数 ν と 熱拡散係数 χ
        nu = alpha * c_s * H_g
        chi = nu
        
        # 3. Paardekooperらの式に基づく各パラメータの計算
        h_gas = H_g / sat.r_orb
        x_s = calculate_x_s(gamma, sat.M_s, M_cp_kg, h_gas)
        P_nu = calculate_P_nu(Omega_K, sat.r_orb, x_s, nu)
        P_chi = calculate_P_chi(Omega_K, sat.r_orb, x_s, chi)
        
        # 4. 新しいマイグレーション係数 β を計算
        b_t1 = calculate_beta(p, q, gamma, P_nu, P_chi)
        
        # 5. 従来の固定値(b_t1)の代わりに beta を使って v_t1 を計算
        v_t1 = calculate_v_t1(b_t1, sat.M_s, M_cp_kg, Sigma_g, sat.r_orb, v_K, c_s)
        v_ad = calculate_v_ad(st_p, eta, v_K)

        # 最終的な移動速度
        v_mig = calculate_v_ad(St_s, eta, v_K) + v_t1
        u_i = calculate_u_i(G, sat.M_s, sat.R_s, delta_v, st_p, Omega_K)
        Ts_strict, _ = calculate_Ts_strict(sat.rho_s, C_p, dRs_dt, M_dot_s, u_i, sat.R_s, T_d)


        
        # 次のステップの値を一時保存
        sat.next_r_orb = sat.r_orb + v_mig * (dt_year * 365.25 * 24 * 3600)
        sat.next_R_s = sat.R_s + dRs_dt * (dt_year * 365.25 * 24 * 3600)
        sat.next_Ts = Ts_strict
        sat.next_M_s = (4.0 / 3.0) * np.pi * sat.next_R_s**3 * sat.rho_s
        sat.next_b_t1 = b_t1
        sat.next_T_d = T_d
        sat.next_P_eff = P_eff
        sat.next_st_p = st_p
        sat.next_u_i = u_i
        sat.next_q = q
        sat.next_P = P
        sat.next_v_ad = v_ad
        sec_per_year = 365.25 * 24 * 3600
        sat.next_t_ad = abs(sat.r_orb / v_ad) / sec_per_year
        # ★ 追加: ボンディ半径の計算
        sat.next_R_B = (G * sat.M_s) / (c_s**2)
        sat.next_M_atm = M_atm
        sat.next_R_out = R_out
        sat.next_R_B = R_B
        sat.next_P_surf = P_surf_cgs

        # --- 追加: 氷質量の獲得 ---
        dM_s = sat.next_M_s - sat.M_s  # このステップで獲得した質量
        
        # スノーライン(T_snow)より外側なら、獲得したペブルの50%が氷だと仮定
        if T_d < T_snow_val:
            sat.next_M_ice = sat.M_ice + dM_s * 0.5
        else:
            sat.next_M_ice = sat.M_ice  # 氷は増えない

    # --- ② 軌道共鳴チェック（内側の衛星から外側へ順番に判定） ---
    active_sats.sort(key=lambda s: s.r_orb)
    
    for i, sat in enumerate(active_sats):
        # イオはキャビティ（空洞の端）で止まる
        if sat.next_r_orb < 5.89 * R_J:
            sat.next_r_orb = 5.89 * R_J
            
     # 2:1軌道共鳴の判定（すぐ内側の衛星の 1.5874倍 の位置で止まる）
        if i > 0:
            resonance_pos = active_sats[i-1].next_r_orb * (2.0**(2/3))
            
            # 追加: 前のステップで既に共鳴に捕獲されていたか（位置が揃っていたか）を判定
            was_in_resonance = (sat.r_orb <= active_sats[i-1].r_orb * (2.0**(2/3)) * 1.001)
            
            # 修正: 内側に食い込むのを防ぐだけでなく、既に捕獲されている場合は軌道を固定する
            if sat.next_r_orb < resonance_pos or was_in_resonance:
                sat.next_r_orb = resonance_pos           
        
        # 状態を確定し、リストに記録
        sat.r_orb, sat.R_s, sat.M_s = sat.next_r_orb, sat.next_R_s, sat.next_M_s
        sat.M_ice = sat.next_M_ice
        sat.M_rock = sat.next_M_rock       
        sat.rho_s = sat.next_rho_s      

        sat.history_t.append(t_year)
        sat.history_Rs.append(sat.R_s / 100000.0)
        sat.history_r.append(sat.r_orb / R_J)
        sat.history_Ts.append(sat.next_Ts)
        sat.history_b_t1.append(sat.next_b_t1)
        sat.history_T_d.append(sat.next_T_d)
        sat.history_P_eff.append(sat.next_P_eff)
        sat.history_st_p.append(sat.next_st_p)
        sat.history_u_i.append(sat.next_u_i)
        sat.history_q.append(sat.next_q)
        sat.history_P.append(sat.next_P)
        sat.history_v_ad.append(sat.next_v_ad)
        sat.history_t_ad.append(sat.next_t_ad)
        sat.history_R_B.append(sat.next_R_B / 100000.0)
        # --- 追加: 大気関連データの記録 ---
        sat.history_M_atm.append(sat.next_M_atm)
        sat.history_R_out.append(sat.next_R_out / 100000.0)  # km単位に変換
        sat.history_P_surf.append(sat.next_P_surf / 10.0)    # ★ 追加: dyn/cm^2 から Pa に変換
        sat.history_ice_frac.append(sat.M_ice / sat.M_s)
    t_year += dt_year

# ==========================================
# 3. 計算終了後、結果を出力・グラフ化
# ==========================================
colors = ['tab:orange', 'tab:cyan', 'tab:blue', 'tab:gray', 'tab:magenta']

# ★ 追加: 未収束の割合をターミナルに表示
print("-" * 40)
print(f"qの総計算回数: {q_total_count} 回")
print(f"うち未収束回数: {q_non_converge_count} 回")
if q_total_count > 0:
    print(f"未収束の割合: {(q_non_converge_count / q_total_count) * 100:.2f} %")
print("-" * 40)

# ---------------------------------------------------
# ★ ここに追加: 最終的な質量と氷質量分率の出力・保存
# ---------------------------------------------------
output_text = "=" * 40 + "\n"
output_text += "計算終了時 (3000万年) の衛星の最終パラメータ\n"
output_text += "=" * 40 + "\n"

for sat in satellites:
    # 質量を g から kg に変換
    final_mass_kg = sat.M_s / 1000.0
    
    # 氷質量分率の計算 (前回追加した M_ice が実装されているか確認)
    if hasattr(sat, 'M_ice'):
        ice_frac = (sat.M_ice / sat.M_s) * 100.0
        ice_str = f"{ice_frac:.1f} %"
    else:
        ice_str = "未計算 (M_iceの実装が必要です)"
        
    output_text += f"[{sat.name}]\n"
    output_text += f"  最終質量   : {final_mass_kg:.3e} kg \n"
    output_text += f"  氷質量分率 : {ice_str}\n"
    output_text += f"  最終軌道   : {sat.r_orb / R_J:.2f} R_J\n"
    output_text += "-" * 40 + "\n"

# ターミナルに表示
print(output_text)

# テキストファイルとして保存
with open("final_satellite_status.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

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
plt.savefig("orbital_evolution_central_flax.png")
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
plt.savefig("growth_history_central_flax.png")
plt.close()

# --- 3枚目: 表面温度 (Thermal Evolution) ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_Rs, sat.history_Ts, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('Surface Temp (K)')
plt.xlabel('衛星半径（km）')
plt.title('表面温度')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("thermal_evolution_central_flax.png")
plt.close()


# --- 4枚目:マイグレーション係数   ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_b_t1, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('Migration Coefficient')
plt.xlabel('Time (Years)')
plt.title('マイグレーション係数')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("migration_coefficient_central_flax.png")
plt.close()

# --- 追加: 円盤ガス温度 (Disk Temperature T_d) ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_T_d, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('Disk Midplane Temp T_d (K)')
plt.xlabel('Time (Years)')
plt.title('各衛星軌道における円盤ガス温度')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("disk_temp_central_flax.png")
plt.close()

# --- 追加: 固定軌道における円盤ガス温度 ---
plt.figure(figsize=(8, 5))
plt.plot(fixed_history_t, fixed_history_Td, color='tab:red', linewidth=2)
plt.ylabel('Disk Midplane Temp T_d (K)')
plt.xlabel('Time (Years)')
plt.title(f'固定軌道 ({target_fixed_r/R_J} R_J) における円盤ガス温度')
plt.xscale('log')
plt.tight_layout()
plt.savefig("fixed_orbit_Td_central_flax.png")
plt.close()

# --- 追加: ぺブル集積効率 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_P_eff, label=sat.name, color=colors[i],  linewidth=2)
plt.ylabel('P_eff')
plt.xlabel('Time (Years)')
plt.title(f'ぺブル集積効率')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("Pebble_effciency_central_flax.png")
plt.close()

# --- 追加: ぺブルのストークス数 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_T_d, sat.history_st_p, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('St_p')
plt.xlabel('衛星軌道のガス円盤温度（K）')
plt.title('ぺブルのストークス数')
#plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("St_p_central_flax.png")


# --- 追加：ぺブルと衛星の相対速度 ---
plt.figure(figsize=(8, 5))  # 個別のグラフのサイズを指定
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_u_i, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('u_i')
plt.xlabel('Time (Years)')
plt.title('ぺブルと衛星の相対速度')
plt.xscale('log')           # 独立したグラフになるのでそれぞれにX軸設定が必要
plt.legend()
plt.tight_layout()
plt.savefig("u_i_central_flax.png")
plt.close()                 # メモリを解放して次のキャンバスへ
"""
"""
# --- 
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_t, sat.history_q, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('q)')
plt.xlabel('Time (Years)')
plt.title('q')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("q_central_flax.png")
plt.close()


# --- 追加: ぺブルのストークス数 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_q, sat.history_st_p, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('St_p')
plt.xlabel('q')
plt.title('ぺブルのストークス数')
#plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("St_p_q_centaral_flax.png")

# --- 追加: 円盤ガス温度とガスの圧力の関係 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        P_pa = [p / 10.0 for p in sat.history_P]
        plt.plot(sat.history_T_d, P_pa, label=sat.name, color=colors[i], linewidth=2)

plt.ylabel('P (Pa)')
plt.xlabel('T_d (K)')
plt.title('各衛星軌道におけるP_gとT_dの関係')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("disk_temp_vs_gas_pressure_central_flax.png")
plt.close()

# --- 追加: 各加熱ソースの比較グラフ ---
plt.figure(figsize=(8, 5))
# 全体の温度（実線）
plt.plot(fixed_history_t, fixed_history_Td, label='Total T_d (総合温度)', color='black', linewidth=2)
# 各成分（点線）
plt.plot(fixed_history_t, fixed_history_T_visc, label='T_visc (粘性加熱)', color='tab:red', linestyle='--')
plt.plot(fixed_history_t, fixed_history_T_irr, label='T_irr (木星からの光)', color='tab:orange', linestyle='--')
plt.plot(fixed_history_t, fixed_history_T_sun, label='T_sun (太陽からの光)', color='tab:blue', linestyle='--')

plt.ylabel('Temperature (K)')
plt.xlabel('Time (Years)')
plt.title(f'各加熱ソースの比較 (固定された軌道)')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("temperature_components_central_flax.png")
plt.close()

# --- 追加: 円盤ガス総質量の推移 ---
plt.figure(figsize=(8, 5))
plt.plot(history_disk_mass_t, history_disk_mass, color='tab:purple', linewidth=2)
plt.ylabel('Disk Mass ($M_J$)')
plt.xlabel('Time (Years)')
plt.title('周木星円盤のガス総質量の時間進化')
plt.xscale('log')
plt.yscale('log')  # 質量も大きく変化するため対数スケールが見やすいです
plt.tight_layout()
plt.savefig("disk_mass_evolution_central_flux.png")
plt.close()

# --- 追加: 円盤ガス流入率の推移 ---
plt.figure(figsize=(8, 5))
# kg/s で記録されているものを、M_J/yr に変換したリストを作成
# (1年は約 3.154e7 秒、木星質量は M_J グラム)

sec_per_year = 365.25 * 24 * 3600 *1.0e6
M_dot_g_MJ_yr = [(val * sec_per_year) / M_J for val in history_M_dot_g]

plt.plot(history_disk_mass_t, M_dot_g_MJ_yr, color='tab:purple', linewidth=2)
plt.ylabel('M_dot_g($M_J$/year)', fontsize=14)
plt.xlabel('Time (Myr)', fontsize=14)
plt.title('gas flux evolution', fontsize=16)
# ★ 追加: X軸の目盛り（位置とテキスト）を明示的に指定する
ticks_positions = [1e6, 3e6, 1e7, 3e7]
ticks_labels = ['1', '3', '10', '30']
plt.xticks(ticks_positions, ticks_labels, fontsize=12)
plt.tight_layout()
plt.savefig("disk_gas_flux_central_flux.png")
plt.close()

# --- 追加: 木星質量の推移 ---
plt.figure(figsize=(8, 5))
plt.plot(history_disk_mass_t, history_M_cp, color='tab:purple', linewidth=2)
plt.ylabel('Jupiter Mass ($M_J$)', fontsize=14)
plt.xlabel('Time (Myr)', fontsize=14)
plt.title('Jupiter Mass evolution', fontsize=16)
# ★ 追加: X軸の目盛り（位置とテキスト）を明示的に指定する
ticks_positions = [1e6, 3e6, 1e7, 3e7]
ticks_labels = ['1', '3', '10', '30']
plt.xticks(ticks_positions, ticks_labels, fontsize=12)
plt.tight_layout()
plt.savefig("jupiter_mass_evolution_central_flux.png")
plt.close()
"""
# --- 追加: 円盤内の軌道半径に対するぺブル移動時間とストークス数 (空間分布) ---
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()  # 右側のY軸（St_p用）を作成

snapshot_times = [1.0e6, 2.0e6]
colors_snap = ['tab:blue', 'tab:orange']

# 1 R_J から 50 R_J まで
r_array_RJ = np.logspace(0, np.log10(50), 100)
r_array = r_array_RJ * R_J

for t_snap, col in zip(snapshot_times, colors_snap):
    M_cp_MJ = calculate_M_cp(t_snap, t_gap)
    M_cp_kg = M_cp_MJ * M_J
    M_dot_g_kg_s = calculate_mdot_g(t_snap, t_gap) * M_J / (365.25 * 24 * 3600)
    M_dot_p_base = x * M_dot_g_kg_s


    t_ad_list = []
    st_p_list = []
    valid_r_RJ = []

    for r_val in r_array:
        p, q = calculate_p_q(M_cp_kg, M_dot_g_kg_s, r_val, alpha, r_gg L_cp)
        T_d, Sigma_g = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, r_val, alpha, r_gg, L_cp, q_guess=q)

        if np.isnan(T_d):
            continue

        local_M_dot_p = M_dot_p_base * 0.5 if T_d >= 160.0 else M_dot_p_base
        eta, _ = calculate_eta(M_cp_kg, M_dot_g_kg_s, r_val, alpha, r_gg)
        st_p = calculate_st_p(p, q, local_M_dot_p, M_dot_g_kg_s, alpha, T_d, M_cp_kg, r_val, eta, frag_limit=False)
        v_K = np.sqrt(G * M_cp_kg / r_val)
        v_r_p = -2.0 * (st_p / (st_p**2 + 1.0)) * eta * v_K

        if v_r_p == 0:
            continue

        t_ad_sec = abs(r_val / v_r_p)
        t_ad_yr = t_ad_sec / (365.25 * 24 * 3600)

        t_ad_list.append(t_ad_yr)
        st_p_list.append(st_p)
        valid_r_RJ.append(r_val / R_J)

    ax1.plot(valid_r_RJ, t_ad_list, label=f't_ad ({t_snap/1e6:.1f} Myr)', color=col, linewidth=2, linestyle='-')
    ax2.plot(valid_r_RJ, st_p_list, label=f'St_p ({t_snap/1e6:.1f} Myr)', color=col, linewidth=2, linestyle=':')

# ★ 修正: 赤点線（plt.axhline）を削除しました

ax1.set_ylabel('Pebble Migration Timescale (Years)', fontsize=14)
ax2.set_ylabel('St_p', fontsize=14)
ax1.set_xlabel('Orbital Radius ($R_J$)', fontsize=14)
plt.title('Pebble Migration Timescale', fontsize=16)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.set_xticks([1, 2, 5, 10, 20, 50])
ax1.set_xticklabels(['1', '2', '5', '10', '20', '50'], fontsize=12)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig("pebble_migration_time_vs_r_central_flux.png")
plt.close()
"""
# --- 追加: ボンディ半径と衛星半径の比較 (大気獲得判定) ---
plt.figure(figsize=(10, 6))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        # 衛星半径 R_s を実線でプロット
        plt.plot(sat.history_t, sat.history_Rs, color=colors[i], linestyle='-', linewidth=2, label=f"{sat.name} (R_s)")
        # ボンディ半径 R_B を点線でプロット
        plt.plot(sat.history_t, sat.history_R_B, color=colors[i], linestyle=':', linewidth=2, label=f"{sat.name} (R_B)")

plt.ylabel('Radius (km)', fontsize=14)
plt.xlabel('Time (Years)', fontsize=14)
plt.title('ボンディ半径と衛星半径の進化 (大気獲得条件: R_B > R_s)', fontsize=16)

# 距離の差が大きいため対数スケールに設定
plt.yscale('log')
plt.xscale('log')

# X軸の目盛りを見やすく整理
ticks_positions = [1e6, 3e6, 1e7, 3e7]
ticks_labels = ['1', '3', '10', '30']
plt.xticks(ticks_positions, ticks_labels, fontsize=12)

# 凡例を外側に配置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("radius_comparison_atmosphere_central_flux.png")
plt.close()

# --- 追加: 衛星の大気質量 (M_atm) と質量比の進化 ---
plt.figure(figsize=(10, 6))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        # 大気質量の絶対値 (g) をプロット
        plt.plot(sat.history_t, sat.history_M_atm, color=colors[i], linestyle='-', linewidth=2, label=f"{sat.name} (Mass [g])")

plt.ylabel('Atmosphere Mass (g)', fontsize=14)
plt.xlabel('Time (Years)', fontsize=14)
plt.title('衛星が獲得した大気質量の進化', fontsize=16)
plt.yscale('log')
plt.xscale('log')
plt.xticks(ticks_positions, ticks_labels, fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("atmosphere_mass_evolution_central_flux.png")
plt.close()

# --- 追加: 衛星の地表面気圧 (P_surf) の進化 ---
plt.figure(figsize=(10, 6))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        # 気圧 (Pa) をプロット
        plt.plot(sat.history_t, sat.history_P_surf, color=colors[i], linestyle='-', linewidth=2, label=f"{sat.name}")

plt.ylabel('Surface Pressure (Pa)', fontsize=14)
plt.xlabel('Time (Years)', fontsize=14)
plt.title('衛星の地表面気圧の進化', fontsize=16)

# 気圧の変化が大きいため対数スケールに設定
plt.yscale('log')
plt.xscale('log')

# 目盛りの設定 (既存のticks_positions, ticks_labelsを利用)
plt.xticks(ticks_positions, ticks_labels, fontsize=12)

# 地球の1気圧 (約101325 Pa) に参考線を引く
plt.axhline(y=101325, color='gray', linestyle='--', linewidth=1, label='1 atm (Earth)')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("surface_pressure_evolution_central_flux.png")
plt.close()

# ---固定軌道でのスノーラインの変化---
plt.figure(figsize=(8, 5))

plt.plot(fixed_history_t, fixed_history_Td, label=r"Disk Temperature (T_d)", color="red", linewidth=2)
plt.plot(fixed_history_t, fixed_history_T_snow, label=r"Water Ice condensation Temperature (T_{con})", color="blue", linestyle="--", linewidth=2)

plt.xlabel("Time [years]", fontsize=14)
plt.ylabel("Temperature [K]", fontsize=14)
plt.title("Evolution of Temperatures at Fixed Orbit", fontsize=16)
plt.xscale("log") 
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig("snowline_central_flux.png")
plt.close()