# 三大典型算法数学基础：反向传播、路径追踪、FFT

> 配套笔记：本文档面向**已具备增删改查（CRUD）能力的编码人员**，梳理 AI、图形学与算法三个领域各一个"地基级"算法（反向传播+梯度下降、路径追踪/蒙特卡洛、快速傅里叶变换）所需的大学乃至硕博数学前置知识。
> 体例沿用《CS336-语言模型数学基础.md》：知识点笔记 → 学习路线 → 自测 → 书籍 → 论文 → 参考仓库，并新增与本库各课程文件的联动索引。
> 公式统一使用 **KaTeX** 语法，可在 Typora / Obsidian / VSCode 预览中直接渲染。

---

## 0. 三个算法一句话定位

| 算法 | 领域 | 一句话 | 核心数学 | 性能/能力红利 |
| --- | --- | --- | --- | --- |
| 反向传播 + 梯度下降 | AI / 机器学习 | 用链式法则倒着算出"每个参数该往哪调"，再沿负梯度走一小步 | 微积分 + 线性代数 + 优化 | 显式逐参数求导 $O(n^2)$ → 反向复用 $O(n)$ |
| 路径追踪（蒙特卡洛光线追踪） | 图形学 | 用随机抽样数值求解渲染方程（一个解不出来的积分方程） | 概率统计 + 多重积分 + 数值分析 | 穷举不可能 → 期望估计 + 误差 $\sim 1/\sqrt{N}$ |
| 快速傅里叶变换 FFT | 算法 / 信号 / 图形 | 利用复数单位根的对称性，把 DFT 从 $O(n^2)$ 降到 $O(n\log n)$ | 复分析 + 多项式 + 分治 | 十亿倍量级加速，20 世纪十大算法之一 |

**共同直觉**：三者都是"换一种数学表示，换来数量级的性能或能力"——与业务代码里"加个索引换来千倍查询速度"是同一种思维。

---

## 1. 知识点笔记

### 1.1 反向传播 + 梯度下降（AI 的引擎）

**算法骨架**：

- 正向：输入逐层算出预测值与损失（误差曲面上的高度）。
- 反向：用链式法则从损失倒推每个参数的偏导数（梯度），复用下游结果，避免重复计算。
- 更新：$\theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}$（沿最陡下坡走步长 $\eta$）。

| 知识点 | 在算法中的用途 | 关键公式 / 概念 |
| --- | --- | --- |
| 导数与偏导数 | 单个参数对损失的影响率 | $\dfrac{\partial \mathcal{L}}{\partial W_{ij}}$ |
| 链式法则 | 反向传播的数学本质 | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial y}\dfrac{\partial y}{\partial x}$ |
| 向量 / 矩阵 / 张量运算 | 一层全连接 $h = \sigma(Wx+b)$ | $(AB)_{ij}=\sum_k A_{ik}B_{kj}$ |
| 梯度 $\nabla f$ | 指向上升最快的方向，取负即最陡下降 | $\nabla f = \left(\dfrac{\partial f}{\partial x_1}, \dots\right)$ |
| 泰勒展开（一阶近似） | 解释为什么小步长能降低损失 | $f(x+\epsilon) \approx f(x) + \nabla f^\top \epsilon$ |
| 激活函数求导 | ReLU 的导数是 0/1 门，sigmoid 注意饱和区 | $\sigma'(x)=\sigma(x)(1-\sigma(x))$ |
| 雅可比矩阵 / 链式法则的矩阵形式 | 逐层梯度公式 | $\delta^{l} = (W^{l+1})^\top \delta^{l+1} \odot \phi'(z^{l})$ |
| 交叉熵与 softmax | 分类损失的标准选择（信息论） | $\mathcal{L}=-\sum_n \log \dfrac{e^{z_{y_n}}}{\sum_j e^{z_j}}$ |
| 凸性 / 鞍点 / 收敛性 | 为什么训练会停在"半山腰"却仍然有效 | 病态条件数、Lipschitz 连续（硕博凸优化） |
| 随机梯度、动量、Adam | 大数据下用小批量估计梯度，二阶动量自适应步长 | $\mathbb{E}[g^2]$ 的指数滑动平均 |

**代码直觉**：反向传播在 PyTorch 中就是 `loss.backward()`，其内部是"计算图上的动态规划"。

### 1.2 路径追踪 / 蒙特卡洛（图形学的引擎）

**渲染方程**（图形学的"麦克斯韦方程组"）：

$$
L_o(x, \omega_o) = L_e(x,\omega_o) + \int_{\Omega} f(x,\omega_i,\omega_o)\, L_i(x,\omega_i)\, \cos\theta_i \, \mathrm{d}\omega_i
$$

含义：出射亮度 = 自发光 + 对半球所有入射方向的积分（BRDF × 入射亮度 × 余弦项）。这是个连续积分方程，两边互相依赖，**解析求解不可行**，只能数值求解——用蒙特卡洛把积分变成随机均值。

| 知识点 | 在算法中的用途 | 关键公式 / 概念 |
| --- | --- | --- |
| 线性代数：点积 / 叉积 / 旋转矩阵 | 光线求交、法线计算、相机变换 | $r(t) = o + t d$ 与球面方程联立 |
| 定积分与半球积分 | 渲染方程就是积分方程 | $\int_\Omega f(\omega)\,\mathrm{d}\omega$，$\mathrm{d}\omega = \sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi$ |
| 概率密度函数 PDF / CDF | 描述"往哪个方向采样更可能" | $\int_\Omega p(\omega)\mathrm{d}\omega = 1$ |
| 期望 / 方差 / 大数定律 | 蒙特卡洛估计的合法性 | $\mathbb{E}[\hat{I}]=I$，$\mathrm{Var}(\hat{I}) = \frac{\sigma^2}{N}$ |
| 逆变换采样 | 从均匀随机数生成任意分布的光线方向 | $X = F^{-1}(u),\ u\sim U(0,1)$ |
| 蒙特卡洛积分 | 核心估计公式 | $\hat{I}=\frac{1}{N}\sum_{i=1}^{N}\dfrac{f(X_i)}{p(X_i)}$ |
| 重要性采样 | 把采样分布偏向贡献大的方向，方差骤降 | 余弦加权 PDF $p(\omega)=\frac{\cos\theta}{\pi}$ |
| 多重要量采样 MIS（硕博） | 组合多种采样策略的工业标准 | 分配权重 $w_i = \frac{n_i p_i}{\sum_j n_j p_j}$（Veach 博士论文） |
| 收敛阶与误差分析 | 噪点 $\sim 1/\sqrt{N}$：要降噪手工不到，只能靠采样设计或降噪后处理 | 中心极限定理（本科数理统计） |
| 测度论 / 辐射度学严格化（博士级） | 渲染方程在数学上的严格定义 | Veach 论文第 3、4 章 |

**代码直觉**：蒙特卡洛估 $\pi$ 与路径追踪是同一算法家族——撒点、判命中、求平均。

```python
N = 1_000_000; hits = 0
for _ in range(N):
    x, y = random(), random()      # 在正方形里均匀撒点
    if x*x + y*y <= 1: hits += 1   # 落入四分之一圆
print(4 * hits / N)                # 期望 = π，误差 ~ 1/√N
```

### 1.3 快速傅里叶变换 FFT（算法 / 信号 / 图形通用的引擎）

**问题**：离散傅里叶变换（DFT）把长度 $n$ 的信号变换到频域，朴素实现是矩阵乘向量，$O(n^2)$。

**核心思想**：把信号看作多项式的系数表示，在 $n$ 次单位根 $\omega_n = e^{-2\pi i/n}$ 的幂处求值。利用折半对称性

$$
\omega_n^{2k} = \omega_{n/2}^{k}, \qquad \omega_n^{k + n/2} = -\omega_n^k
$$

把奇偶下标拆成两个规模减半的子问题，递归求解后用"蝴蝶操作"合并，$T(n) = 2T(n/2) + O(n) \Rightarrow O(n\log n)$。

| 知识点 | 在算法中的用途 | 关键公式 / 概念 |
| --- | --- | --- |
| 三角函数与和差化积 | 理解频域分量从哪来 | $\cos(\alpha\pm\beta)=\cos\alpha\cos\beta \mp \sin\alpha\sin\beta$ |
| 复数与欧拉公式 | 旋转的代数化，DFT 的算子就是旋转 | $e^{i\theta}=\cos\theta + i\sin\theta$ |
| 多项式两种表示 | 系数表示 vs 点值表示，卷积在点值表示下是逐点乘法（$O(n)$） | $n$ 个点唯一确定 $n{-}1$ 次多项式 |
| $n$ 次单位根 | FFT 折半合并的钥匙 | $\omega_n^n=1$，$\omega_n^{n/2}=-1$ |
| 分治与主定理 | 递推复杂度 | $T(n)=2T(n/2)+O(n)=O(n\log n)$ |
| 采样定理（奈奎斯特） | 为什么采样率必须高于 2 倍最高频率（CD 取 44.1kHz） | $\mathrm{sinc}$ 插值、频谱混叠 |
| 环论 / 本原单位根（硕博级） | NTT：在模素数 $p$（$p = k\cdot 2^m + 1$）下做"整数精确 FFT"，用于密码学与竞赛卷积 | $g^{-\frac{p-1}{n}} \bmod p$ 当作 $\omega_n$ |
| 数值稳定性（硕博级） | 浮点 FFT 的舍入误差分析，高精度场景用 NTT 替代 | 误差 $\sim O(\log n\cdot\varepsilon)$ |

**代码直觉**：递归 FFT 约 20 行，核心只有"偶/奇分半"与"$\pm t$ 只算一次"。

```python
def fft(a):                          # a: 系数列表，长度为 2 的幂
    if len(a) == 1: return a
    even, odd = fft(a[0::2]), fft(a[1::2])
    n, wn = len(a), cmath.exp(-2j * cmath.pi / len(a))
    y = [0] * n
    for k in range(n // 2):
        t = wn**k * odd[k]
        y[k]         = even[k] + t
        y[k + n//2]  = even[k] - t   # 关键：两个子式中 ±t 共享一次乘法
    return y
```

---

## 2. 学习路线（Roadmap）

共 5 周，"够用即可，遇缺再补"，三个算法各占一周到一周半。

### 阶段 1：反向传播（Week 1–2）

1. 3Blue1Brown《微积分的本质》《线性代数的本质》补直觉。
2. 微积分：链式法则、偏导数、梯度（对应本库 `02-本科/01-数学分析.md`、`02-高等代数.md`）。
3. 手推两层全连接网络的反向传播公式（对照本文件 1.1 节公式），再用 NumPy 实现，最后用 `torch.autograd` 验证。
4. **产出**：不使用任何框架，写出一个会学习的玩具神经网络。

### 阶段 2：路径追踪与蒙特卡洛（Week 3–4）

1. 概率：期望、方差、PDF/CDF、大数定律（对应 `02-本科/05-概率论.md`、`06-数理统计.md`）。
2. 写出蒙特卡洛估 $\pi$，理解误差 $\sim 1/\sqrt{N}$。
3. 读 *Ray Tracing in One Weekend*（免费三部曲），写出一个有噪点的路径追踪器。
4. **产出**：能解释渲染方程每一项的含义，并用自己的渲染器渲染一个金属球。

### 阶段 3：FFT（Week 5）

1. 高中复数 → 欧拉公式 → 单位根（对应 `01-高中人教版/07-复数.md`、`02-本科/07-复变函数.md`）。
2. 手写递归 FFT，并用它做一次多项式乘法（卷积）与大数乘法。
3. 对照 CLRS 第 30 章消化"蝴蝶操作"，有时间再读 NTT 与模素数本原根。

### 阶段 4（硕博级深入，可选）

- 优化收敛性 / 鞍点逃离：`03-硕博/10-凸优化与最优控制.md`。
- MIS 采样与辐射度学严格化：Veach 博士论文，配套 `03-硕博/09-随机分析与马尔可夫过程.md`。
- NTT 与卷积环结构：`03-硕博/06-交换代数与同调代数.md`（环论视角）。

---

## 3. 检测知识掌握程度的材料（Self-Check）

### 3.1 反向传播（概念 → 代码 → 推导）

1. 为什么分类任务用交叉熵损失而不是均方误差？
2. 手推：$Y = XW$，$\mathcal{L}=\|Y-T\|_2^2$，求 $\partial \mathcal{L}/\partial W$。
3. ReLU 在 0 点不可导，PyTorch 如何处理？（次梯度约定）
4. 学习率过大时损失爆炸，用泰勒展开解释原因。
5. 代码自测：纯 NumPy 实现两层 MLP 的前向 + 反向，与 `autograd` 对比梯度误差 $< 10^{-7}$。

### 3.2 路径追踪（概念 → 代码 → 推导）

1. 写出渲染方程，并解释式中每项的含义。
2. 为什么误差是 $1/\sqrt{N}$ 而不是 $1/N$？哪个定律保证的？
3. 余弦加权重要性采样比均匀采样方差低多少？自己设计实验验证。
4. MIS 解决什么问题：一个交集贡献既适合镜面采样又适合面积采样时怎么办？
5. 代码自测：蒙特卡洛估 $\pi$（N=100w，误差 < 0.5%）；实现逆变换采样生成余弦半球方向。

### 3.3 FFT（概念 → 代码 → 推导）

1. 写出 DFT 的矩阵形式，说明朴素实现为什么是 $O(n^2)$。
2. 由欧拉公式推出 $\omega_n^n = 1$ 与 $\omega_n^{n/2} = -1$。
3. 归纳证明：FFT 递归正确且复杂度为 $O(n\log n)$。
4. 4×4 蝴蝶操作图手画一遍，标出每次乘法用到的 $\omega$ 的幂。
5. 代码自测：手写递归 FFT 与 `numpy.fft.fft` 全通误差 $< 10^{-9}$；用 FFT 完成 10 万位大数乘法。

### 3.4 掌握度评级

| 等级 | 标准 |
| --- | --- |
| L1 入门 | 能说清每个算法"在哪一步用了哪个数学对象" |
| L2 应用 | 能照模板跑通或复现官方实现 |
| L3 推导 | 能手推梯度公式、推导蒙特卡洛无偏性、证明 FFT 递归式 |
| L4 迁移 | 能做重要性采样调优、写出 NTT、实现 Adam 变体 |

---

## 4. 书籍（Books）

**微积分与线性代数（反向传播前置）**
- 3Blue1Brown《微积分的本质》《线性代数的本质》（视频，直觉优先）
- Gilbert Strang, *Introduction to Linear Algebra*（配 MIT 公开课）
- Goodfellow et al., *Deep Learning*（花书第 2–4、6 章，接线正式）

**概率与数值积分（路径追踪前置）**
- Sheldon Ross, *A First Course in Probability*（概率基础）
- Wasserman, *All of Statistics*（速通统计，配合 CLT）
- Matt Pharr et al., *Physically Based Rendering*（PBRT，免费在线，工业级路径追踪）

**复分析与算法（FFT 前置）**
- Ahlfors, *Complex Analysis*（选读前三章即可够用，配合 `02-本科/07-复变函数.md`）
- Cormen et al., *Introduction to Algorithms*（CLRS）第 30 章《多项式与 FFT》
- Proakis, *Digital Signal Processing*（采样定理与频域视角）

**硕博级延伸**
- Boyd & Vandenberghe, *Convex Optimization*（优化收敛性）
- Eric Veach, *Robust Monte Carlo Methods for Light Transport Simulation*（博士论文，MIS 起点）
- von zur Gathen & Gerhard, *Modern Computer Algebra*（FFT / NTT 的代数深度）

---

## 5. 论文（Papers）

按与三个算法的关联度排序：

- **LeCun et al., 1988/1998, *Gradient-Based Learning Applied to Document Recognition*** — 反向传播 + 卷积网络的工业奠基（LeNet）
- **Rumelhart, Hinton & Williams, 1986, *Learning representations by back-propagating errors*** — 反向传播的原始出处（Nature）
- **Kingma & Ba, 2015, *Adam: A Method for Stochastic Optimization*** — 自适应步长的标准优化器（arXiv:1412.6980）
- **Kajiya, 1986, *The Rendering Equation*** (SIGGRAPH) — 渲染方程的提出，路径追踪的数学出发点（DOI: 10.1145/15922.15902）
- **Veach & Guibas, 1995/1997, *Optimally Combining Sampling Techniques for Monte Carlo Rendering*** — MIS 多重要量采样
- **Cooley & Tukey, 1965, *An Algorithm for the Machine Calculation of Complex Fourier Series*** — FFT 的开山之作（DOI: 10.1090/S0025-5718-1965-0178586-1）
- **Gauss 1805（未发表手稿）/ Heideman et al., 1984, *Gauss and the history of the FFT*** — 历史注记：高斯早于 Cooley–Tukey 160 年已有类似思想

> 阅读建议：每读一篇，回到第 1 节标注该论文用到的数学点，形成"论文—数学"双向索引。

---

## 6. GitHub 参考仓库（Reference Repos）

- **karpathy/nanoGPT**、**karpathy/llm.c** — 最小 GPT 与纯 C 实现，逐行看反向传播
- **ddbourgin/numpy-ml** — 纯 NumPy 各类模型，手推梯度对照训练
- **RayTracing/raytracing.github.io** — *Ray Tracing in One Weekend* 三部曲的配套代码与勘误
- **mmp/pbrt-v4** — PBRT 官方实现（ Importance/MIS 的工业范本）
- **atcoder/ac-library** — 内含经竞赛验证的 NTT 卷积实现模板
- **numpy/numpy**（`numpy.fft`）、**fftw/fftw** — 工业级 FFT 的两种工程境界
- 本库 `03-硕博/papers/` 下的论文清单与 `CS336-语言模型数学基础.md`（反向传播相关数学的展开阅读）

---

## 附：与本库课程文件的联动索引

| 本文件知识点 | 建议回查的本库课程文件 |
| --- | --- |
| 链式法则 / 梯度 / 泰勒展开 | `02-本科/01-数学分析.md`、`01-高中人教版/15-一元函数的导数及其应用.md` |
| 矩阵乘法 / 雅可比 / SVD | `02-本科/02-高等代数.md` |
| 交叉熵 / 期望 / 大数定律 | `02-本科/05-概率论.md`、`01-高中人教版/10-概率.md` |
| CLT / 误差 $1/\sqrt{N}$ | `02-本科/06-数理统计.md` |
| 半球积分 / 多重积分 | `02-本科/01-数学分析.md`（多元积分部分） |
| 优化收敛 / 鞍点 / 凸性 | `03-硕博/10-凸优化与最优控制.md`、`02-本科/11-数值分析.md` |
| 复数 / 欧拉公式 / 单位根 | `01-高中人教版/07-复数.md`、`02-本科/07-复变函数.md` |
| NTT / 本原根 / 环结构 | `02-本科/09-抽象代数.md`、`03-硕博/06-交换代数与同调代数.md` |
| 采样随机过程视角 | `02-本科/12-随机过程.md`、`03-硕博/09-随机分析与马尔可夫过程.md` |
| 三角函数 / 和差化积 | `01-高中人教版/05-三角函数.md` |
