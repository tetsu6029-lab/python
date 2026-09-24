import numpy as np
import matplotlib.pyplot as plt

# tauの範囲を設定（対数スケールで0.01から100まで）
tau = np.logspace(-2, 2, 500)

# 係数gの計算
g_tau = ((3.0 / 8.0) * tau + 1.0 / (4.8 * tau))**(0.25)

# グラフの描画
plt.figure(figsize=(8, 6))

# メインの関数のプロット（漸近線を削除）
plt.plot(tau, g_tau, color='black', linewidth=2)

# 最小値の計算とプロット
tau_min = np.sqrt(8.0 / (3.0 * 4.8)) # 約0.745
g_min = ((3.0 / 8.0) * tau_min + 1.0 / (4.8 * tau_min))**(0.25) # 約0.863

# 最小値のポイントを打つ
plt.scatter(tau_min, g_min, color='green', zorder=5)

# 最小値の座標をグラフ上にテキスト表示（縦軸の値gも追加）
plt.annotate(f'Minimum\n($\\tau$ = {tau_min:.2f}, g = {g_min:.2f})', 
             xy=(tau_min, g_min), 
             xytext=(0, 20), 
             textcoords='offset points', 
             ha='center', 
             color='green',
             fontsize=12)

# 軸の設定
plt.xscale('log') # 横軸は対数スケールのまま
# 縦軸の目盛りを明示的に設定（0.5から3.0まで0.5刻み）
plt.yticks(np.arange(0.5, 3.5, 0.5)) 

# 軸ラベルとタイトル（数式は見栄え良くTeXフォーマットに調整）
plt.xlabel(r'Optical Depth $\tau$', fontsize=14)
plt.ylabel(r'Coefficient $g(\tau)$', fontsize=14)
plt.title('Coefficient g(tau) ', fontsize=16)

# グリッドの表示
plt.grid(True, which="both", ls="--", alpha=0.5)

# グラフの表示
plt.tight_layout()
plt.show()
