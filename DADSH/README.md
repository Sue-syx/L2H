**Objective Function：**
$$
\begin{align} \mathop{min}\limits_{U,V,W_1,W_2} F = \rVert U^T&V-\frac{2}{3}lS \rVert_F^2    	+\varphi\rVert \tilde{\\V}-U\rVert_F^2  \\    +\eta \left \| UI\right \|_F^2&+\mu\left\|UU^T-\frac{2}{3}nI\right\|_F^2    +\gamma\Psi(U) \tag{1.1}  \\      \Psi(U)= \sum ^{l}_{k=1}\sum &^{m}_{i=1}2\left| U_{ki}\right| \cdot \left( 1-U^{2}_{ki}\right) \\U\in \left[ -1,1\right]^{l\times m},&V\in \left\{ -1,0,1\right\} ^{l\times n},\tilde V\in \left\{ -1,0,1\right\} ^{l\times m},S\in\left\{-1,1\right\} ^{m\times n} \\\end{align}
$$
c：类别数，l：编码长度，m、n：样本数

U：网络输出，V：数据集编码

**更新V**
$$
\begin{align}
&\mathop{min}\limits_{V}
  F(V) = \rVert U^TV-\frac{2}{3}lS \rVert_F^2  
  	+\varphi\rVert \tilde{\\V}-U\rVert_F^2  \tag{2.1}\\
& Define\;\ \bar{U}={\bar{u}_{j=1}^n},
	where\;\ \bar{u}_j\;\ is\;\ defined\;\ as\;\ follows: 
	\bar{u}_j=\left\{
		\begin{aligned}
			u_j &,\ \ if\ \ j\in\Omega \\
            0&,\ \ else
		\end{aligned}
		\right. \\
& F =\left\| VU^{T}\right\| ^{2}_{F}
	-2Tr(V[\frac{2}{3}cU^TS+\varphi \bar U^T])
	+const \tag{2.2} \\ 
& Let\ \ Q=-2(\frac{2}{3}cS^TU+\varphi \bar{U}) \\
& \mathop{min}\limits_{V}F(V) =\rVert VU^{T}\rVert_F^2
									   +Tr(VQ^T) +const \\
& s.t.\ \ V\in \{-1,0,1\}^{n\times l} \tag{2.2} \\ \\

 
& \ \ F(V_{*k}) = Tr(V_{*k}[2U_{*k}^T\hat{U}_k\hat{V}^T_k+Q_{*k}^T])+const\\
& \ \ s.t.\ \ V_{*k}\in {-1,0,1}^n \\

& \therefore V_{*k}=-trisign(2\hat{V}_{k}\hat{U}^T_kU_{*k}+Q_{*k}) \tag{2.3} \\
& trisign(x)=\begin{cases}-1,\ \ x\leq -0.33\\ 
						  0,\ \ -0.33<x\leq 0.33\\ 
						  1,\ \ 0.33\leq x\end{cases}

\end{align}
$$
