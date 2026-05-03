# 劣駆動2重振り子のモデリングから線形近似システムまで

## 0. 目的

劣駆動な2重振り子について、次の流れを数式として整理する。

1. 座標を定義する。
2. 重心位置と速度を求める。
3. 運動エネルギー $T$ とポテンシャルエネルギー $U$ を作る。
4. ラグランジアン $L=T-U$ から非線形運動方程式を導く。
5. Rayleigh の散逸関数で粘性摩擦を入れる。
6. 非線形運動方程式を

    $$
    M(q)\ddot q=h(q,\dot q,u)
    $$

    の形にまとめる。

7. 静止平衡点まわりで線形近似システム

    $$
    \dot x=A_cx+B_cu
    $$

    を作る。

---

## 1. 記号と座標

一般化座標を

$$
q=\begin{bmatrix}\theta_1\\ \theta_2\end{bmatrix}
$$

とする。

状態は

$$
x=\begin{bmatrix}\theta_1\\ \theta_2\\ \dot\theta_1\\ \dot\theta_2\end{bmatrix}
$$

とする。

座標の取り方は次の通り。

- $\theta_1$: 第1リンクの下向き鉛直からの角度。
- $\theta_2$: 第2リンクの上向き鉛直からの角度。
- 第2関節の相対角速度は

$$
\dot\phi=\dot\theta_1+\dot\theta_2
$$

である。

ここが重要である。今回の座標では第2関節の相対角は $\theta_1+\theta_2$ に対応する。

パラメータは次のように置く。

$$
M_1,\ M_2: \text{リンク質量}
$$

$$
L_1: \text{第1リンク全長},\qquad l_1: \text{第1リンク支点から重心までの距離}
$$

$$
l_2: \text{第2関節から第2リンク重心までの距離}
$$

$$
J_1,\ J_2: \text{各リンク重心まわりの慣性モーメント}
$$

$$
b_1: \text{第1関節の粘性摩擦係数}
$$

$$
b_2: \text{第2関節の相対角速度に対する粘性摩擦係数}
$$

$$
ku: \text{第1関節に加わる入力トルク}
$$

---

## 2. 位置ベクトル

定数平行移動は速度と運動方程式に影響しないので、位置の定数項は必要に応じて捨てる。

第1リンク重心を

$$
P_1=\begin{bmatrix}
l_1\sin\theta_1\\
-l_1\cos\theta_1
\end{bmatrix}
$$

と置く。

第2関節位置を

$$
P_2=\begin{bmatrix}
L_1\sin\theta_1\\
-L_1\cos\theta_1
\end{bmatrix}
$$

と置く。

第2リンク重心は

$$
P_3=P_2+\begin{bmatrix}
l_2\sin\theta_2\\
l_2\cos\theta_2
\end{bmatrix}
$$

である。

したがって

$$
P_3=\begin{bmatrix}
L_1\sin\theta_1+l_2\sin\theta_2\\
-L_1\cos\theta_1+l_2\cos\theta_2
\end{bmatrix}
$$

となる。

---

## 3. 速度ベクトル

第1リンク重心の速度は

$$
\dot P_1=\begin{bmatrix}
l_1\dot\theta_1\cos\theta_1\\
l_1\dot\theta_1\sin\theta_1
\end{bmatrix}
$$

なので、

$$
\|\dot P_1\|^2=l_1^2\dot\theta_1^2
$$

である。

第2リンク重心の速度は

$$
\dot P_3=\begin{bmatrix}
L_1\dot\theta_1\cos\theta_1+l_2\dot\theta_2\cos\theta_2\\
L_1\dot\theta_1\sin\theta_1-l_2\dot\theta_2\sin\theta_2
\end{bmatrix}
$$

である。

したがって、

$$
\|\dot P_3\|^2
=
\left(L_1\dot\theta_1\cos\theta_1+l_2\dot\theta_2\cos\theta_2\right)^2
+
\left(L_1\dot\theta_1\sin\theta_1-l_2\dot\theta_2\sin\theta_2\right)^2
$$

を展開する。

$$
\|\dot P_3\|^2
=
L_1^2\dot\theta_1^2\cos^2\theta_1
+2L_1l_2\dot\theta_1\dot\theta_2\cos\theta_1\cos\theta_2
+l_2^2\dot\theta_2^2\cos^2\theta_2
$$

$$
\qquad
+
L_1^2\dot\theta_1^2\sin^2\theta_1
-2L_1l_2\dot\theta_1\dot\theta_2\sin\theta_1\sin\theta_2
+l_2^2\dot\theta_2^2\sin^2\theta_2
$$

$$
\cos^2\theta_1+
\sin^2\theta_1=1
$$

$$
\cos^2\theta_2+
\sin^2\theta_2=1
$$

$$
\cos\theta_1\cos\theta_2-\sin\theta_1\sin\theta_2
=\cos(\theta_1+\theta_2)
$$

より、

$$
\boxed{
\|\dot P_3\|^2
=
L_1^2\dot\theta_1^2
+2L_1l_2\dot\theta_1\dot\theta_2\cos(\theta_1+\theta_2)
+l_2^2\dot\theta_2^2
}
$$

となる。

ここで $\cos(\theta_1+\theta_2)$ が出るのは、今回の座標定義による。

---

## 4. 運動エネルギー

第1リンクの運動エネルギーは

$$
T_1=\frac12 M_1\|\dot P_1\|^2+\frac12 J_1\dot\theta_1^2
$$

なので、

$$
T_1=\frac12(M_1l_1^2+J_1)\dot\theta_1^2
$$

である。

第2リンクの運動エネルギーは

$$
T_2=\frac12 M_2\|\dot P_3\|^2+\frac12 J_2\dot\theta_2^2
$$

である。

したがって、

$$
T_2
=\frac12M_2L_1^2\dot\theta_1^2
+M_2L_1l_2\dot\theta_1\dot\theta_2\cos(\theta_1+\theta_2)
+\frac12(M_2l_2^2+J_2)\dot\theta_2^2
$$

となる。

よって全運動エネルギーは

$$
T=T_1+T_2
$$

であり、

$$
\boxed{
T=\frac12\left(J_1+M_1l_1^2+M_2L_1^2\right)\dot\theta_1^2
+M_2L_1l_2\dot\theta_1\dot\theta_2\cos(\theta_1+\theta_2)
+\frac12\left(J_2+M_2l_2^2\right)\dot\theta_2^2
}
$$

となる。

---

## 5. ポテンシャルエネルギー

重力ポテンシャルは鉛直座標に比例する。

第1リンク重心の鉛直座標は定数を除いて

$$
-l_1\cos\theta_1
$$

である。

第2リンク重心の鉛直座標は定数を除いて

$$
-L_1\cos\theta_1+l_2\cos\theta_2
$$

である。

したがって、

$$
U=-M_1gl_1\cos\theta_1
+M_2g\left(-L_1\cos\theta_1+l_2\cos\theta_2\right)
$$

である。

整理して、

$$
\boxed{
U=-(M_1l_1+M_2L_1)g\cos\theta_1
+M_2l_2g\cos\theta_2
}
$$

となる。

ここで、第1リンクは $\theta_1=0$ が下向き安定姿勢なので $-\cos\theta_1$ 型になる。第2リンクは $\theta_2=0$ が上向き倒立姿勢なので $+\cos\theta_2$ 型になる。

---

## 6. ラグランジアン

ラグランジアンは

$$
L=T-U
$$

である。

後の計算を短くするため、次の定数を置く。

$$
\alpha=J_1+M_1l_1^2+M_2L_1^2
$$

$$
\beta=M_2L_1l_2
$$

$$
\gamma=J_2+M_2l_2^2
$$

$$
d=(M_1l_1+M_2L_1)g
$$

$$
e=M_2l_2g
$$

$$
S=\theta_1+\theta_2
$$

このとき、

$$
T=\frac12\alpha\dot\theta_1^2+\beta\dot\theta_1\dot\theta_2\cos S+\frac12\gamma\dot\theta_2^2
$$

$$
U=-d\cos\theta_1+e\cos\theta_2
$$

なので、

$$
L=\frac12\alpha\dot\theta_1^2+\beta\dot\theta_1\dot\theta_2\cos S+\frac12\gamma\dot\theta_2^2+d\cos\theta_1-e\cos\theta_2
$$

である。

---

## 7. 散逸関数と一般化力

第1関節には粘性摩擦 $b_1$ がある。

第2関節には、相対角速度

$$
\dot\phi=\dot\theta_1+
\dot\theta_2
$$

に対する粘性摩擦 $b_2$ がある。

Rayleigh の散逸関数を

$$
\boxed{
R=\frac12 b_1\dot\theta_1^2+\frac12b_2(\dot\theta_1+
\dot\theta_2)^2
}
$$

と置く。

粘性摩擦による一般化力は

$$
Q_i^{\mathrm{damp}}=-\frac{\partial R}{\partial \dot q_i}
$$

である。

したがって、

$$
Q_1^{\mathrm{damp}}
=-\frac{\partial R}{\partial\dot\theta_1}
=-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)
$$

$$
Q_2^{\mathrm{damp}}
=-\frac{\partial R}{\partial\dot\theta_2}
=-b_2(\dot\theta_1+
\dot\theta_2)
$$

である。

第1関節に入力トルク $ku$ が加わるので、入力による一般化力は

$$
Q^{\mathrm{input}}=\begin{bmatrix}ku\\0\end{bmatrix}
$$

である。

したがって、全一般化力は

$$
\boxed{
Q_1=ku-b_1\dot\theta_1-b_2(\dot\theta_1+\dot\theta_2)
}
$$

$$
\boxed{
Q_2=-b_2(\dot\theta_1+\dot\theta_2)
}
$$

となる。

第2関節摩擦は第2式だけでなく第1式にも入る。理由は、相対角 $\phi$ が $\theta_1$ と $\theta_2$ の両方に依存するからである。

---

## 8. Euler-Lagrange 方程式

Euler-Lagrange 方程式は

$$
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta_i}
-\frac{\partial L}{\partial\theta_i}
=Q_i
$$

である。

---

## 9. $\theta_1$ についての計算

まず、

$$
\frac{\partial L}{\partial\dot\theta_1}
=\alpha\dot\theta_1+\beta\dot\theta_2\cos S
$$

である。

時間微分すると、

$$
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta_1}
=\alpha\ddot\theta_1+\beta\left(\ddot\theta_2\cos S-
\dot\theta_2(\dot\theta_1+\dot\theta_2)\sin S\right)
$$

である。

次に、

$$
\frac{\partial L}{\partial\theta_1}
=-\beta\dot\theta_1\dot\theta_2\sin S-d\sin\theta_1
$$

である。

したがって、

$$
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta_1}
-\frac{\partial L}{\partial\theta_1}
$$

は、

$$
\alpha\ddot\theta_1+\beta\ddot\theta_2\cos S-\beta\dot\theta_2(\dot\theta_1+
\dot\theta_2)\sin S+\beta\dot\theta_1\dot\theta_2\sin S+d\sin\theta_1
$$

である。

速度積の項を整理する。

$$
-\beta\dot\theta_2(\dot\theta_1+
\dot\theta_2)\sin S+\beta\dot\theta_1\dot\theta_2\sin S
=-\beta\dot\theta_2^2\sin S
$$

したがって、

$$
\alpha\ddot\theta_1+\beta\cos S\,\ddot\theta_2-\beta\sin S\,\dot\theta_2^2+d\sin\theta_1=Q_1
$$

である。

$$
Q_1=ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)
$$

を代入すると、

$$
\alpha\ddot\theta_1+\beta\cos S\,\ddot\theta_2-\beta\sin S\,\dot\theta_2^2+d\sin\theta_1
=ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)
$$

である。

加速度項だけを左辺に残すと、

$$
\boxed{
\alpha\ddot\theta_1+\beta\cos S\,\ddot\theta_2
=ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin S\,\dot\theta_2^2-d\sin\theta_1
}
$$

となる。

---

## 10. $\theta_2$ についての計算

まず、

$$
\frac{\partial L}{\partial\dot\theta_2}
=\beta\dot\theta_1\cos S+
\gamma\dot\theta_2
$$

である。

時間微分すると、

$$
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta_2}
=\beta\left(\ddot\theta_1\cos S-
\dot\theta_1(\dot\theta_1+
\dot\theta_2)\sin S\right)+
\gamma\ddot\theta_2
$$

である。

次に、

$$
\frac{\partial L}{\partial\theta_2}
=-\beta\dot\theta_1\dot\theta_2\sin S+e\sin\theta_2
$$

である。

したがって、

$$
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta_2}
-\frac{\partial L}{\partial\theta_2}
$$

は、

$$
\beta\ddot\theta_1\cos S-
\beta\dot\theta_1(\dot\theta_1+
\dot\theta_2)\sin S+
\gamma\ddot\theta_2+
\beta\dot\theta_1\dot\theta_2\sin S-e\sin\theta_2
$$

である。

速度積の項を整理する。

$$
-\beta\dot\theta_1(\dot\theta_1+
\dot\theta_2)\sin S+
\beta\dot\theta_1\dot\theta_2\sin S
=-\beta\dot\theta_1^2\sin S
$$

したがって、

$$
\beta\cos S\,\ddot\theta_1+
\gamma\ddot\theta_2-
\beta\sin S\,\dot\theta_1^2-e\sin\theta_2=Q_2
$$

である。

$$
Q_2=-b_2(\dot\theta_1+
\dot\theta_2)
$$

を代入すると、

$$
\beta\cos S\,\ddot\theta_1+
\gamma\ddot\theta_2-
\beta\sin S\,\dot\theta_1^2-e\sin\theta_2
=-b_2(\dot\theta_1+
\dot\theta_2)
$$

である。

加速度項だけを左辺に残すと、

$$
\boxed{
\beta\cos S\,\ddot\theta_1+
\gamma\ddot\theta_2
=-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin S\,\dot\theta_1^2+e\sin\theta_2
}
$$

となる。

---

## 11. 非線形運動方程式の行列表現

以上より、

$$
M(q)\ddot q=h(q,\dot q,u)
$$

の形で書ける。

質量行列は

$$
\boxed{
M(q)=\begin{bmatrix}
\alpha & \beta\cos(\theta_1+	heta_2)\\
\beta\cos(\theta_1+	heta_2)&\gamma
\end{bmatrix}
}
$$

である。

右辺は

$$
\boxed{
h(q,\dot q,u)=\begin{bmatrix}
ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin(\theta_1+	heta_2)\dot\theta_2^2-d\sin\theta_1\\
-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin(\theta_1+	heta_2)\dot\theta_1^2+e\sin\theta_2
\end{bmatrix}
}
$$

である。

すなわち、

$$
\boxed{
\begin{bmatrix}
\alpha & \beta\cos(\theta_1+	heta_2)\\
\beta\cos(\theta_1+	heta_2)&\gamma
\end{bmatrix}
\begin{bmatrix}
\ddot\theta_1\\
\ddot\theta_2
\end{bmatrix}
=
\begin{bmatrix}
ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin(\theta_1+	heta_2)\dot\theta_2^2-d\sin\theta_1\\
-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin(\theta_1+	heta_2)\dot\theta_1^2+e\sin\theta_2
\end{bmatrix}
}
$$

である。

この形が、非線形シミュレーションと線形化の分岐点になる。

---

## 12. 非線形状態方程式

状態を

$$
x=\begin{bmatrix}
\theta_1\\
\theta_2\\
\dot\theta_1\\
\dot\theta_2
\end{bmatrix}
$$

とする。

このとき、

$$
\dot x=\begin{bmatrix}
\dot\theta_1\\
\dot\theta_2\\
\ddot\theta_1\\
\ddot\theta_2
\end{bmatrix}
$$

である。

非線形運動方程式から、

$$
\ddot q=M(q)^{-1}h(q,\dot q,u)
$$

なので、

$$
\boxed{
\dot x=f(x,u)=\begin{bmatrix}
\dot q\\
M(q)^{-1}h(q,\dot q,u)
\end{bmatrix}
}
$$

である。

実装では $M(q)^{-1}$ を明示的に作るより、線形方程式

$$
M(q)\ddot q=h(q,\dot q,u)
$$

を解く方がよい。

---

## 13. 線形化する平衡点

ここでは、平衡点を

$$
q_0=\begin{bmatrix}0\\0\end{bmatrix},\qquad
\dot q_0=\begin{bmatrix}0\\0\end{bmatrix},\qquad
u_0=0
$$

とする。

この平衡点は、

$$
\theta_1=0,
\qquad
\theta_2=0
$$

すなわち、第1リンクが下向き、第2リンクが上向きの姿勢である。

平衡点では

$$
\dot q_0=0,
\qquad
\ddot q_0=0
$$

である。

---

## 14. 線形化の基本方針

非線形運動方程式は

$$
M(q)\ddot q=h(q,\dot q,u)
$$

である。

平衡点まわりの微小変化を

$$
q=q_0+\delta q
$$

$$
\dot q=\dot q_0+\delta\dot q
$$

$$
\ddot q=\ddot q_0+\delta\ddot q
$$

$$
u=u_0+\delta u
$$

と書く。

今回の平衡点では

$$
q_0=0,
\qquad
\dot q_0=0,
\qquad
\ddot q_0=0,
\qquad
u_0=0
$$

なので、

$$
q=\delta q,
\qquad
\dot q=\delta\dot q,
\qquad
\ddot q=\delta\ddot q,
\qquad
u=\delta u
$$

として読める。

左辺を考える。

$$
M(q)\ddot q
=M(q_0+\delta q)\delta\ddot q
$$

である。

$$
M(q_0+\delta q)=M(q_0)+O(\delta q)
$$

なので、

$$
M(q_0+\delta q)\delta\ddot q
=M(q_0)\delta\ddot q+O(\delta q\,\delta\ddot q)
$$

である。

$$
\delta q\,\delta\ddot q
$$

は2次の微小量なので、一次近似では捨てる。

したがって、左辺は

$$
\boxed{
M_0\delta\ddot q
}
$$

になる。

ここで、

$$
M_0=M(q_0)
$$

である。

右辺は普通に一次近似して、

$$
h(q,\dot q,u)
\simeq
h(q_0,\dot q_0,u_0)
+
H_q\delta q
+
H_{\dot q}\delta\dot q
+
H_u\delta u
$$

である。

平衡点では

$$
h(q_0,\dot q_0,u_0)=0
$$

なので、

$$
h(q,\dot q,u)
\simeq
H_q\delta q
+
H_{\dot q}\delta\dot q
+
H_u\delta u
$$

である。

したがって、線形化された2階の運動方程式は

$$
\boxed{
M_0\delta\ddot q
=H_q\delta q+H_{\dot q}\delta\dot q+H_u\delta u
}
$$

となる。

---

## 15. $M_0$ の計算

平衡点では

$$
\theta_1=0,
\qquad
\theta_2=0
$$

なので、

$$
\cos(\theta_1+	heta_2)=1
$$

である。

したがって、

$$
\boxed{
M_0=M(q_0)=
\begin{bmatrix}
\alpha&\beta\\
\beta&\gamma
\end{bmatrix}
}
$$

である。

元のパラメータで書くと、

$$
\boxed{
M_0=
\begin{bmatrix}
J_1+M_1l_1^2+M_2L_1^2 & M_2L_1l_2\\
M_2L_1l_2 & J_2+M_2l_2^2
\end{bmatrix}
}
$$

である。

---

## 16. $H_q$ の計算

$$
H_q=\left.\frac{\partial h}{\partial q}\right|_{\rm eq}
$$

である。

まず、

$$
h_1=ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin S\,\dot\theta_2^2-d\sin\theta_1
$$

である。

$$
\frac{\partial h_1}{\partial\theta_1}
=\beta\cos S\,\dot\theta_2^2-d\cos\theta_1
$$

である。

平衡点では

$$
\dot\theta_2=0,
\qquad
\theta_1=0
$$

なので、

$$
\left.\frac{\partial h_1}{\partial\theta_1}\right|_{\rm eq}
=-d
$$

である。

また、

$$
\frac{\partial h_1}{\partial\theta_2}
=\beta\cos S\,\dot\theta_2^2
$$

なので、平衡点では

$$
\left.\frac{\partial h_1}{\partial\theta_2}\right|_{\rm eq}=0
$$

である。

次に、

$$
h_2=-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin S\,\dot\theta_1^2+e\sin\theta_2
$$

である。

$$
\frac{\partial h_2}{\partial\theta_1}
=\beta\cos S\,\dot\theta_1^2
$$

なので、平衡点では

$$
\left.\frac{\partial h_2}{\partial\theta_1}\right|_{\rm eq}=0
$$

である。

また、

$$
\frac{\partial h_2}{\partial\theta_2}
=\beta\cos S\,\dot\theta_1^2+e\cos\theta_2
$$

なので、平衡点では

$$
\left.\frac{\partial h_2}{\partial\theta_2}\right|_{\rm eq}=e
$$

である。

したがって、

$$
\boxed{
H_q=
\begin{bmatrix}
-d&0\\
0&e
\end{bmatrix}
}
$$

である。

元のパラメータで書くと、

$$
\boxed{
H_q=
\begin{bmatrix}
-(M_1l_1+M_2L_1)g&0\\
0&M_2l_2g
\end{bmatrix}
}
$$

である。

第1リンクの重力項は戻す符号である。第2リンクの重力項は倒立姿勢から倒れる符号である。

---

## 17. $H_{\dot q}$ の計算

$$
H_{\dot q}=\left.\frac{\partial h}{\partial \dot q}\right|_{\rm eq}
$$

である。

まず、

$$
h_1=ku-b_1\dot\theta_1-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin S\,\dot\theta_2^2-d\sin\theta_1
$$

なので、

$$
\frac{\partial h_1}{\partial\dot\theta_1}=-(b_1+b_2)
$$

であり、

$$
\frac{\partial h_1}{\partial\dot\theta_2}=-b_2+2\beta\sin S\,\dot\theta_2
$$

である。

平衡点では

$$
S=0,
\qquad
\dot\theta_2=0
$$

なので、

$$
\left.\frac{\partial h_1}{\partial\dot\theta_2}\right|_{\rm eq}=-b_2
$$

である。

次に、

$$
h_2=-b_2(\dot\theta_1+
\dot\theta_2)+\beta\sin S\,\dot\theta_1^2+e\sin\theta_2
$$

なので、

$$
\frac{\partial h_2}{\partial\dot\theta_1}=-b_2+2\beta\sin S\,\dot\theta_1
$$

である。

平衡点では

$$
S=0,
\qquad
\dot\theta_1=0
$$

なので、

$$
\left.\frac{\partial h_2}{\partial\dot\theta_1}\right|_{\rm eq}=-b_2
$$

である。

また、

$$
\frac{\partial h_2}{\partial\dot\theta_2}=-b_2
$$

である。

したがって、

$$
\boxed{
H_{\dot q}=
\begin{bmatrix}
-(b_1+b_2)&-b_2\\
-b_2&-b_2
\end{bmatrix}
}
$$

である。

---

## 18. $H_u$ の計算

$$
H_u=\left.\frac{\partial h}{\partial u}\right|_{\rm eq}
$$

である。

$$
h_1=ku+\cdots
$$

$$
h_2=\cdots
$$

なので、

$$
\boxed{
H_u=
\begin{bmatrix}
k\\0
\end{bmatrix}
}
$$

である。

---

## 19. 線形化された2階系

ここまでより、

$$
\boxed{
M_0\delta\ddot q
=H_q\delta q+H_{\dot q}\delta\dot q+H_u\delta u
}
$$

である。

すなわち、

$$
\boxed{
\begin{bmatrix}
\alpha&\beta\\
\beta&\gamma
\end{bmatrix}
\begin{bmatrix}
\delta\ddot\theta_1\\
\delta\ddot\theta_2
\end{bmatrix}
=
\begin{bmatrix}
-d&0\\
0&e
\end{bmatrix}
\begin{bmatrix}
\delta\theta_1\\
\delta\theta_2
\end{bmatrix}
+
\begin{bmatrix}
-(b_1+b_2)&-b_2\\
-b_2&-b_2
\end{bmatrix}
\begin{bmatrix}
\delta\dot\theta_1\\
\delta\dot\theta_2
\end{bmatrix}
+
\begin{bmatrix}
k\\0
\end{bmatrix}
\delta u
}
$$

である。

---

## 20. 線形状態方程式

線形化状態を

$$
\delta x=\begin{bmatrix}
\delta q\\
\delta\dot q
\end{bmatrix}
=
\begin{bmatrix}
\delta\theta_1\\
\delta\theta_2\\
\delta\dot\theta_1\\
\delta\dot\theta_2
\end{bmatrix}
$$

とする。

すると、

$$
\delta\dot x
=\begin{bmatrix}
\delta\dot q\\
\delta\ddot q
\end{bmatrix}
$$

である。

2階系より、

$$
\delta\ddot q=M_0^{-1}H_q\delta q+M_0^{-1}H_{\dot q}\delta\dot q+M_0^{-1}H_u\delta u
$$

なので、

$$
\boxed{
\delta\dot x
=
\begin{bmatrix}
0_{2\times2}&I_2\\
M_0^{-1}H_q&M_0^{-1}H_{\dot q}
\end{bmatrix}
\delta x
+
\begin{bmatrix}
0_{2\times1}\\
M_0^{-1}H_u
\end{bmatrix}
\delta u
}
$$

である。

したがって、

$$
\boxed{
A_c=
\begin{bmatrix}
0_{2\times2}&I_2\\
M_0^{-1}H_q&M_0^{-1}H_{\dot q}
\end{bmatrix}
}
$$

$$
\boxed{
B_c=
\begin{bmatrix}
0_{2\times1}\\
M_0^{-1}H_u
\end{bmatrix}
}
$$

である。

平衡点がゼロなので、実用上は $x$ と $\delta x$ を同一視できる。ただし厳密には線形化変数は $\delta x$ である。

---

## 21. 明示形

$$
M_0=
\begin{bmatrix}
\alpha&\beta\\
\beta&\gamma
\end{bmatrix}
$$

なので、

$$
\Delta=\alpha\gamma-\beta^2
$$

と置けば、

$$
M_0^{-1}=\frac1\Delta
\begin{bmatrix}
\gamma&-\beta\\
-\beta&\alpha
\end{bmatrix}
$$

である。

したがって、

$$
M_0^{-1}H_q
=\frac1\Delta
\begin{bmatrix}
\gamma&-\beta\\
-\beta&\alpha
\end{bmatrix}
\begin{bmatrix}
-d&0\\
0&e
\end{bmatrix}
$$

$$
\boxed{
M_0^{-1}H_q
=\frac1\Delta
\begin{bmatrix}
-\gamma d&-\beta e\\
\beta d&\alpha e
\end{bmatrix}
}
$$

である。

また、

$$
M_0^{-1}H_{\dot q}
=\frac1\Delta
\begin{bmatrix}
\gamma&-\beta\\
-\beta&\alpha
\end{bmatrix}
\begin{bmatrix}
-(b_1+b_2)&-b_2\\
-b_2&-b_2
\end{bmatrix}
$$

より、

$$
\boxed{
M_0^{-1}H_{\dot q}
=\frac1\Delta
\begin{bmatrix}
-\gamma(b_1+b_2)+\beta b_2&-\gamma b_2+\beta b_2\\
\beta(b_1+b_2)-\alpha b_2&\beta b_2-\alpha b_2
\end{bmatrix}
}
$$

である。

さらに、

$$
M_0^{-1}H_u
=\frac1\Delta
\begin{bmatrix}
\gamma&-\beta\\
-\beta&\alpha
\end{bmatrix}
\begin{bmatrix}
k\\0
\end{bmatrix}
$$

より、

$$
\boxed{
M_0^{-1}H_u
=\frac1\Delta
\begin{bmatrix}
\gamma k\\
-\beta k
\end{bmatrix}
}
$$

である。

したがって、明示的な線形化状態方程式は

$$
\boxed{
\delta\dot x=A_c\delta x+B_c\delta u
}
$$

であり、

$$
\boxed{
A_c=
\begin{bmatrix}
0&0&1&0\\
0&0&0&1\\
-\gamma d/\Delta&-\beta e/\Delta&\{-\gamma(b_1+b_2)+\beta b_2\}/\Delta&\{-\gamma b_2+\beta b_2\}/\Delta\\
\beta d/\Delta&\alpha e/\Delta&\{\beta(b_1+b_2)-\alpha b_2\}/\Delta&\{\beta b_2-\alpha b_2\}/\Delta
\end{bmatrix}
}
$$

$$
\boxed{
B_c=
\begin{bmatrix}
0\\
0\\
\gamma k/\Delta\\
-\beta k/\Delta
\end{bmatrix}
}
$$

である。

ただし、実装では $M_0^{-1}$ を明示的に作るより、

$$
M_0X=H_q
$$

$$
M_0Y=H_{\dot q}
$$

$$
M_0z=H_u
$$

を解く方がよい。

---

## 22. チェックポイント

質量行列は対称である。

$$
M(q)^T=M(q)
$$

第1リンクの重力線形化は安定側なので負符号である。

$$
H_{q,11}=-d
$$

第2リンクの重力線形化は倒立側なので正符号である。

$$
H_{q,22}=e
$$

第2関節摩擦は相対角速度に対する摩擦なので、

$$
-b_2(\dot\theta_1+\dot\theta_2)
$$

が両方の運動方程式に現れる。

静止平衡点まわりでは、

$$
\sin S\,\dot\theta_i^2
$$

のような速度二乗項は一次線形化では消える。

静止平衡点まわりでは、

$$
M(q)\ddot q
$$

の $M(q)$ の変化分は

$$
O(\delta q\,\delta\ddot q)
$$

になり、一次近似では消える。

したがって、線形化では

$$
M_0\delta\ddot q
=H_q\delta q+H_{\dot q}\delta\dot q+H_u\delta u
$$

だけを見ればよい。

---

## 23. 最終まとめ

このモデリングの流れは、

$$
\text{位置ベクトル}
\rightarrow
T,U
\rightarrow
L=T-U
\rightarrow
\text{散逸関数}
\rightarrow
M(q)\ddot q=h(q,\dot q,u)
\rightarrow
\dot x=f(x,u)
\rightarrow
\delta\dot x=A_c\delta x+B_c\delta u
$$

である。

今回の最重要点は、非線形運動方程式を

$$
\boxed{
M(q)\ddot q=h(q,\dot q,u)
}
$$

の形に持っていくことである。

この形は、非線形シミュレーションにも線形化にも使える。

