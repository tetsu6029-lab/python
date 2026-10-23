import numpy as np
import matplotlib.pyplot as plt

# 光学厚み tau の範囲を設定 (例: 10^-2 から 10^2)
tau = np.logspace(-2, 2, 500)

# 画像の数式に基づく関数 g の計算
g = ((3.0 / 8.0) * tau + 1.0 / (4.8 * tau)) ** (1.0 / 4.0)

# グラフの描画
plt.figure(figsize=(8, 5))
plt.plot(tau, g, linewidth=2, color='tab:blue')

# 軸とラベルの設定
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Optical Depth $\tau$', fontsize=14)
plt.ylabel(r'$g(\tau)$', fontsize=14)
plt.title(r'$g = \left(\frac{3}{8}\tau + \frac{1}{4.8\tau}\right)^{1/4}$', fontsize=16)

# グリッドを表示
plt.grid(True, which="both", linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig("g_function.png")
plt.show()