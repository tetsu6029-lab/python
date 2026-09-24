from scipy.optimize import newton, brentq
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import newton

plt.rcParams['font.family'] = 'IPAexGothic'

# --- 物理定数 ---
G = 6.6743e-8             # 万有引力定数 (cm^3/g/s^2)
sigma_sb = 5.67e-5         # ステファン・ボルツマン定数 (W/cm^2/K^4)
r_gg = 1.7e-7              # 粒子とガスの面密度比
k_B = 1.380649e-16         # ボルツマン定数 (erg/K)
m_g = 3.9e-24              # ガスの平均分子質量 (g)
alpha = 1.0e-4              # 乱流の強さ
M_J = 1.90e30              # 木星質量（g）
t_gap = 1.0e6              # ギャップ形成時の時間（yr）
R_J = 7.15e9               # 木星半径（cm）
x = 0.0026                 # ダストとガスの降着比
C_D = 0.5                  # 抵抗係数 
C_p = 1.4e7               # 比熱 (erg/g/K)
gamma = 1.4
t_gap =1.0e6
# ==========================================
# 事前準備：物理量を計算する関数群
# ==========================================

# 時間t[yr]における中心惑星質量 M_cp [M_J] を計算する関数
def calculate_M_cp(t, t_gap):
    t_end = 3.0e6
    M_dot_g_gap = 1.0e-6
    
    # 逆算したt=0での木星初期質量（t=100万年で元の0.4 M_J になるように設定）
    M_initial = 0.1 
    
    if t <= t_gap:
        # 0〜100万年: 一定ペースで質量増加
        return M_initial + M_dot_g_gap * t
    elif t <= t_end:
        # 100万〜300万年: 減衰しながら質量増加（積分による厳密解）
        M_at_gap = M_initial + M_dot_g_gap * t_gap
        delta_M = (M_dot_g_gap / (2.0 * (t_end - t_gap))) * ((t_end - t_gap)**2 - (t_end - t)**2)
        return M_at_gap + delta_M
    else:
        # 300万年以降は木星の成長も完全にストップ
        M_at_gap = M_initial + M_dot_g_gap * t_gap
        delta_M_max = (M_dot_g_gap / 2.0) * (t_end - t_gap) # ピッタリ 0.3 M_J になります
        return M_at_gap + delta_M_max

# 衛星質量 M_s と 中心惑星質量 M_cp から質量比 mu_s を計算する関数
def calculate_mu_s(M_s, M_cp): 
    return M_s / M_cp

# ガスのフラックス (M_dot_g) を計算する関数(300万年で完全に消失するモデル)
def calculate_mdot_g(t, t_gap):
    t_end = 3.0e6
    M_dot_g_gap = 1.0e-6
    
    if t <= t_gap:
        M_dot_g = M_dot_g_gap
    elif t < t_end:
        M_dot_g = M_dot_g_gap * (t_end - t) / (t_end - t_gap)
    else:
        # 修正: 300万年以降でガスがマイナスにならないように極小値を設定
        M_dot_g = 1e-20
        
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
        # ニュートン法の代わりにブレント法（brentq）を使用し、1e-5 K から 10000.0 K の間で解を探索
        T_d_solution = brentq(balance_equation, 1e-5, 10000.0)
    except (ValueError, RuntimeError):
        print(f"警告: 温度計算が収束しませんでした。(r={r/R_J:.2f} R_J)")
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

    # スノーラインを境に限界衝突速度(v_cr)を変更
    if T_d >= 160.0:
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

# ニュートン法による厳密解
def calculate_Ts_strict(rho_s, C_p, dRs_dt, M_dot_s, u_i, R_s, T_d):
    heating_flux = (M_dot_s * u_i**2) / (8 * np.pi * R_s**2)
    T_s_guess = ((heating_flux / sigma_sb) + T_d**4)**0.25
    
    heat_capacity_term = rho_s * C_p * dRs_dt
    constant_cooling_term = heating_flux + sigma_sb * T_d**4 + heat_capacity_term * T_d
    
    def energy_balance(T_s):
        return sigma_sb * T_s**4 + heat_capacity_term * T_s - constant_cooling_term
        
    def energy_balance_prime(T_s):
        return 4.0 * sigma_sb * T_s**3 + heat_capacity_term
    
    try:
        T_s_strict = newton(energy_balance, x0=T_s_guess, fprime=energy_balance_prime)
    except RuntimeError:
        T_s_strict = T_s_guess
        
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
        
        # 記録用リスト
        self.history_t, self.history_Rs, self.history_r, self.history_Ts = [], [], [], []
        self.history_b_t1, self.history_T_d, self.history_P_eff, self.history_st_p = [], [], [], []
        self.history_u_i, self.history_q, self.history_P = [], [], []
# シミュレーションの初期設定
t_year = 1.0e6             # イオが捕獲される時間からスタート
dt_year = 1000.0            # タイムステップ
t_max_year = 3.0e7         # 3000万年

# 追加: 固定軌道での温度を記録するための準備
target_fixed_r = 14.8 * R_J  # グラフ化したい軌道半径を指定
fixed_history_t = []
fixed_history_Td = []

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
  
  # 追加: 指定した固定軌道におけるT_dを計算して記録
    Td_fixed, _ = calculate_disk_structure(M_cp_kg, M_dot_g_kg_s, target_fixed_r, alpha, r_gg)
    fixed_history_t.append(t_year)
    fixed_history_Td.append(Td_fixed)

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
       
        eta, H_g = calculate_eta(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg)
        p, q = calculate_p_q(M_cp_kg, M_dot_g_kg_s, sat.r_orb, alpha, r_gg)
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
        
        dRs_dt = M_dot_s / (4 * np.pi * sat.rho_s * sat.R_s**2)
        
        c_s = np.sqrt((k_B * T_d) / m_g)
        rho_g = Sigma_g / (np.sqrt(2.0 * np.pi) * H_g)
        P =rho_g * c_s**2
        St_s = calculate_satellite_stokes_number(C_D, sat.rho_s, sat.R_s, rho_g, eta, v_K, Omega_K)
        
        
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
    t_year += dt_year

# ==========================================
# 3. 計算終了後、結果を出力・グラフ化
# ==========================================
colors = ['tab:orange', 'tab:cyan', 'tab:blue', 'tab:gray', 'tab:magenta']

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
plt.savefig("orbital_evolution_t_dep.png")
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
plt.savefig("growth_history_t_dep.png")
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
plt.savefig("thermal_evolution_t_dep.png")
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
plt.savefig("migration_coefficient_t_dep.png")
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
plt.savefig("disk_temp_t_dep.png")
plt.close()

# --- 追加: 固定軌道における円盤ガス温度 ---
plt.figure(figsize=(8, 5))
plt.plot(fixed_history_t, fixed_history_Td, color='tab:red', linewidth=2)
plt.ylabel('Disk Midplane Temp T_d (K)')
plt.xlabel('Time (Years)')
plt.title(f'固定軌道 ({target_fixed_r/R_J} R_J) における円盤ガス温度')
plt.xscale('log')
plt.tight_layout()
plt.savefig("fixed_orbit_Td_t_dep.png")
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
plt.savefig("Pebble_effciency_t_dep.png")
plt.close()

# --- 追加: ぺブルのストークス数 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_T_d, sat.history_st_p, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('St_p')
plt.xlabel('衛星軌道のガス円盤温度（K）')
plt.title('ぺブルのストークス数')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("St_p_t_dep.png")
plt.close()

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
plt.savefig("u_i_t_dep.png")
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
plt.savefig("q_t_dep.png")
plt.close()


# --- 追加: ぺブルのストークス数 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        plt.plot(sat.history_q, sat.history_st_p, label=sat.name, color=colors[i], linewidth=2)
plt.ylabel('St_p')
plt.xlabel('q')
plt.title('ぺブルのストークス数')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("St_p_q_t_dep.png")

# --- 追加: 円盤ガス温度とガスの圧力の関係 ---
plt.figure(figsize=(8, 5))
for i, sat in enumerate(satellites):
    if len(sat.history_t) > 0:
        P_pa = [p / 10.0 for p in sat.history_P]
        plt.plot(P_pa, sat.history_T_d, label=sat.name, color=colors[i], linewidth=2)

plt.ylabel(' T_d (K)')
plt.xlabel('P (Pa)')
plt.title('各衛星軌道におけるT_dとP_gの関係')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("disk_temp_vs_gas_pressure_t_dep.png")
plt.close()



