import numpy as np
import matplotlib.pyplot as plt

# グラフのフォント設定（既存コードに合わせています）
plt.rcParams['font.family'] = 'IPAexGothic'

# 比較する温度の範囲: 120 K から 220 K (500分割)
T = np.linspace(120, 220, 500)

# ==========================================
# 1. 従来のシンプルな式 (Shibaike et al. 2019 等で想定される近似式)
# 出力単位: dyn/cm^2
# ==========================================
P_sat_classic_cgs = np.exp(-6070.0 / T + 30.86)
P_sat_classic = P_sat_classic_cgs / 10

# ==========================================
# 2. Murphy-Koop Equation (2005) 
# 画像の式。元の出力単位は Pa (パスカル)
# ==========================================
P_sat_MK_Pa = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)



# ==========================================
# グラフの描画
# ==========================================
plt.figure(figsize=(9, 6))

# 両モデルのプロット
plt.plot(T, P_sat_classic, label=" Bauer et al. 1997", 
         color='tab:red', linewidth=2, linestyle='--')
plt.plot(T, P_sat_MK_Pa, label="Murphy-Koop 2005", 
         color='tab:blue', linewidth=2)


# グラフの装飾
plt.yscale('log')  # 圧力が指数関数的に変化するため対数スケール
plt.xlabel('Temperature [K]', fontsize=14)
plt.ylabel('Saturation Vapor Pressure [Pa]', fontsize=14)
plt.title('水氷の飽和蒸気圧モデルの比較', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig("compare_vapor_pressure.png")
