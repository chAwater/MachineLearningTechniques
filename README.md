# MachineLearningTechniques
My Notebooks for Machine Learning Techniques (by @hsuantien)

---

### 目录

---

### Coursera Links

- [机器学习基石上 (Machine Learning Foundations)-Mathematical Foundations](https://www.coursera.org/learn/ntumlone-mathematicalfoundations)
- [机器学习基石下 (Machine Learning Foundations)-Algorithmic Foundations](https://www.coursera.org/learn/ntumlone-algorithmicfoundations)
- [机器学习基石 Notebooks](https://github.com/chAwater/MachineLearningFoundations)
- 机器学习技法（已下架，请移步油管或其他资源）
- by [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)

### 前言介绍

《机器学习技法》是国立台湾大学资讯工程系的 **林轩田** 老师开设的课程（**中文授课**）。

该课程旨在延续《机器学习基石》，包括机器学习的 **哲学**、关键 **理论** 和核心 **技术** 。

《机器学习技法》更进一步的关注在 **特征转换** 的一些相关技术，让学生能够更专业的了解和使用机器学习。

- 需要先学习《机器学习基石》（下简称《基石》）再学习《机器学习技法》

### 其他支持

- [<img class="emoji" title=":atom:" alt=":atom:" src="https://github.githubassets.com/images/icons/emoji/atom.png" height="20" width="20" align="absmiddle"> Atom](https://atom.io)
- [CodeCogs (LaTeX Editor API)](http://latex.codecogs.com)
- [Grip -- GitHub Readme Instant Preview](https://github.com/joeyespo/grip)
- [Markdown Toc](https://github.com/nok/markdown-toc)

---

## Lecture 1: Linear Support Vector Machine

—— 介绍边界的概念

—— 介绍支持向量机、名字的由来

—— 介绍支持向量机的一般解法和理论保证

### 最大边界的线性分类器

先回顾一下线性分类问题（参见《基石》）：
- <img src="http://latex.codecogs.com/svg.latex?{h(\mathbf{x})=\mathrm{sign}(\mathbf{w}^T\mathbf{x})}"/>
<!-- - ![](https://render.githubusercontent.com/render/math?math=h\(\mathbf{x}\)=\mathrm{sign}\(\mathbf{w}^T\mathbf{x}\)) -->
- PLA 算法

对于一些问题，可能存在多种 PLA 的解都可以把数据分开，得到这些解取决于把数据放入 PLA 算法的顺序，并且看上去“一样好”。

![](./Snapshot/Snap01.png)

但是我们 **人类** 会倾向于选择最右边的这个，这是因为当存在一些噪音的时候（比如测试数据和训练数据之间存在一些误差）最右边的这个线可以容忍最多的噪音、误差。

所以我们希望 **每个点都和我们的线距离最远**，也可以说我们希望我们能够找到一个 **最胖** 的线，这个线离它最近的点的距离最远。
这个线有多“胖”，就是说这个线的 **边界 (margin)** 有多大。

总结一下：
- 找到一个可以正确区分数据的线性分类器（超平面）
- 得到每个数据和这个线性分类器的距离，取最小的距离作为边界
- 最大化这个边界

![](./Snapshot/Snap02.png)

### 简化问题

首先，需要把线性分类问题中的 <b>w</b><sub>0</sub> 单独拿出来讨论，称之为 b，并把 <b>x</b><sub>0</sub> 从原先的 <b>x</b> 中去掉：
<img src="http://latex.codecogs.com/svg.latex?{h(\mathbf{x})=\mathrm{sign}(\mathbf{w}^T\mathbf{x}+b)}"/>
<!-- ![](https://render.githubusercontent.com/render/math?math=h\(\mathbf{x}\)=\mathrm{sign}\(\mathbf{w}^T\mathbf{x}%2Bb\)) -->

---

[Issues #1] 为什么要把 b 拿出来？

---

#### 简化距离

对于每个数据点和分类器（超平面）之间的距离：
- 考虑超平面上的任意一点 x<sup>'</sup>
- **w** 与超平面上任意一点的乘积为 0 ，因此 **w** 相当于超平面的法向量（<img src="http://latex.codecogs.com/svg.latex?{\mathbf{w}^T\mathbf{x}'+b=0}"/>）
<!-- （![](https://render.githubusercontent.com/render/math?math=\mathbf{w}^T\mathbf{x}'%2Bb=0)） -->

- 数据点和超平面的距离，相当于数据点和连接的向量在垂直于超平面方向（ w ）上的投影
- 因为这数据点可以被分类器（超平面）区分，因此有 <img src="http://latex.codecogs.com/svg.latex?{\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)>0}"/>
<!-- ![](https://render.githubusercontent.com/render/math?math=\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)\gt0) -->

所以距离可以简化为：

<img src="http://latex.codecogs.com/svg.latex?{\textrm{distance}(\mathbf{x},b,\mathbf{w})=|\frac{\mathbf{w}^T}{\|\mathbf{w}\|}(\mathbf{x}-\mathbf{x}')|=\frac{1}{\|\mathbf{w}\|}|\mathbf{w}^T+b|=\frac{1}{\|\mathbf{w}\|}\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)}"/>

<!-- ![](https://render.githubusercontent.com/render/math?math=\mathrm{distance}\(\mathbf{x},b,\mathbf{w}\)=\left|\frac{\mathbf{w}^T}{\|\mathbf{w}\|}\(\mathbf{x}-\mathbf{x}'\)\right|=\frac{1}{\|\mathbf{w}\|}|\mathbf{w}^T%2Bb|=\frac{1}{\|\mathbf{w}\|}\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)) -->

#### 简化条件 - 向量缩放和边界的定义

对于表示这个分类器（超平面）的向量来说，向量的缩放（改变长度不改变方向）并不影响，仍然可以表示这个超平面。

因此，我们可以对这个向量进行一个 **特殊的缩放**，使：
<img src="http://latex.codecogs.com/svg.latex?{\min_{n=1,\dots,N}\,\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)=1}"/>

<!-- ![](https://render.githubusercontent.com/render/math?math=\underset{n=1\,\\!\\!...\,\\!\\!N}{\min}\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)=1) -->

这样操作之后有两个好处：
1. 显然在这种缩放下可以保证 <img src="http://latex.codecogs.com/svg.latex?{\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)>0}"/>，因此这个条件可以去掉；
2. 边界变成 <img src="http://latex.codecogs.com/svg.latex?{\frac{1}{\|\mathbf{w}\|}}"/>

所以这个问题就被简化为：

<img src="http://latex.codecogs.com/svg.latex?{\max_{b,\mathbf{w}}\frac{1}{\|\mathbf{w}\|}\textrm{\quad\,subject\,to\quad\,}\,\min_{n=1,\dots,N}\,\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)=1}"/>

#### 简化条件 - 有帮助的宽松

我们继续简化这个问题，我们先将条件 **放宽** 到：对于所有的数据（所有的 n ）有 <img src="http://latex.codecogs.com/svg.latex?{\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)\,\ge\,1}"/>

如果 **大于等于** 中的 **等于** 对于宽松后的解不成立，那么这个 **放宽** 后的问题与之前的问题是不同的。

但是，如果等于不成立，那么我们必然可以找到一个新的 **缩放** 使得 **等于** 成立，并且这个新的 **缩放** 是比原来的 **缩放** 程度更大，因此 || **w** || 只会更小，因此我们能找到一个更大的 **边界**，这就与宽松后的解产生了矛盾，因此这个宽松中的等于是必然成立的。

然后我们通过取倒数将最大化问题转成最小化问题，并用平方去掉绝对值（根号），再加上一个常数项。

![](./Snapshot/Snap03.png)

### 支持向量机

这种算法被称为 **支持向量机 (Support Vector Machine, SVM)**，是因为在超平面`边界`上的那些数据点决定了这个超平面和边界，而其他地方的数据点对于边界和超平面来说是不必要的。
这些在超平面边界上的点被称为`支持向量`（的候选），因为这些点就好像在支撑着这个超平面一样。

```
####### 感受数学的力量吧！！！ #######
```

那么我们继续来求解这个问题，这个问题有一些特性：
- 这个问题是`凸的二次函数`
- 这个问题是 **w** 和 b 的`线性运算`

具有这种特性的问题被称为 **二次规划 (Quadratic Programming, QP)**，有很多现成的工具来求解这种问题，那么我们只要把这个问题转化成标准二次规划问题的形式就很好处理了。（作为“文科生”，这段转化忽略...）

![](./Snapshot/Snap04.png)

这就是硬边界（hard-margin，每个数据都是正确区分的）的线性（非线性变换后线性也可以）支持向量机的标准解法。

---

### 支持向量机的理论保证

首先我们来比较一下`正则化`（参见《基石》）和 SVM 在最小化和条件上的区别：

|                |              Minimize               |                 Constraint                 |
|:--------------:|:-----------------------------------:|:------------------------------------------:|
| Regularization |         <i>E</i><sub>in</sub>       | <b>w</b><sup><i>T</i></sup><b>w</b> &le; C |
|       SVM      | <b>w</b><sup><i>T</i></sup><b>w</b> |   <i>E</i><sub>in</sub>=0 [and scaling]    |

所以 SVM 和正则化有些类似。

---

另外，我们再来讨论一下 SVM 的 `VC Dimension`（参见《基石》）。首先来讨论一下 `Dichotomy`（参见《基石》），当没有`边界`的时候，一个线性分类器在一些数据上可以 `Shatter`（参见《基石》）。

但是当我们加入了`边界`之后，在一些情况下可能就不能`Shatter`了，因为原先能够`Shatter`的分类器没有足够大的`边界`。这样就相当于减少了 `Dichotomy` 也就是减少了 `VC Dimension`，因此有更好的 `Generalization`。

这种方法相当于再数据层面增加了一些条件来控制`VC Dimension`。

![](./Snapshot/Snap05.png)

---
---
---

## Lecture 2: Dual Support Vector Machine

—— 介绍 SVM 的对偶问题

—— 介绍拉格朗日函数

### 回顾 SVM 和对偶问题（对偶 SVM）

在上面我们介绍了 SVM 的标准解法（转化成`二次规划`问题），并且我们可以利用`特征转换`把 **x** 转换到 **z** 空间中，这样在`VC`上我们就可以做到有很高的复杂度，但又因为`边界`限制而没有那么多的复杂度。

但是还有一个问题，因为 QP 有 <i>d</i>+1 个变量，当我们使用的特征转换很强大的时候，转换后的 <i>d</i> 就会很大，因此这个二次规划问题也很难解。我们希望解 SVM 的过程能够和  <i>d</i> 没关系，比如夸张点 <i>d</i> 无穷大也能解。

那么下面我们就来将原来的 SVM 求解中的有 <i>d</i>+1 个变量和 N 个条件 的 QP，转换成它的 **对偶问题**，这新的问题只有 N 个变量和 N+1 个条件。这里将有很多很多的数学（之前的还不算多...），我们会只介绍概念和重点。

我们用到了一个很重要的工具就是 **拉格朗日乘数 Lagrange Multiplier**，常用于解决有条件的最佳化问题。在《基石》，我们在做`正则化`的时候有用到，将正则化的条件放进最小化的问题中。

不一样的地方是，在正则化的时候，&lambda; 是限制条件常数 C 的一种代替，因此也是常数。

而在 SVM 中，我们把 &lambda; 当做一个变量来代替原本 SVM 的变量来解，这就是对偶问题。因为 SVM 有 N 个条件，所以就有 N 个 &lambda;。

---

[Issues #2] 什么是对偶问题？

---

原来的 SVM 问题：

<img src="http://latex.codecogs.com/svg.latex?{\min_{b,\mathbf{w}}\,\frac{1}{2}\mathbf{w}^T\mathbf{w}\textrm{\,\,s.\,t.\,\,}\,\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b)\,\ge\,1\,\textrm{for}\,n=1,\dots,N}"/>

我们构建一个 **拉格朗日函数** （其中的 a<sub>n</sub> 相当于 &lambda;<sub>n</sub>）：

<img src="http://latex.codecogs.com/svg.latex?{\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})=\underbrace{\frac{1}{2}\mathbf{w}^T\mathbf{w}}_{\textrm{objective}}\;+\;\sum_{n=1}^N\alpha_n\underbrace{(1-\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b))}_{\textrm{constraint}}}"/>

这样一来有：

<img src="http://latex.codecogs.com/svg.latex?{\textrm{SVM}\,\equiv\,\min_{b,\mathbf{w}}\left(\max_\mathrm{{all}\,\alpha_n\,\ge\,0}\,\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})\right)}"/>

这是因为：
- 限制 &alpha; 大于 0
- 当条件不满足时，`条件项`大于 0，最大化拉格朗日函数会得到无穷大
- 当条件满足时，`条件项`小于等于 0，最大化拉格朗日函数等于`目标项`
- 对`目标项`最小化就等同于原来的 SVM 问题

### 拉格朗日对偶 SVM

在 SVM 的对偶问题中，对于某一个特定的 &alpha;<sup>'</sup>，拉格朗日函数的值必定小于等于最大的那一个：

<img src="http://latex.codecogs.com/svg.latex?{\min_{b,\mathbf{w}}\left(\max_\mathrm{{all}\,\alpha_n\,\ge\,0}\,\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})\right)\ge\min_{b,\mathbf{w}}\,\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha}')}"/>

对不等式右边取最大化，不等式仍然成立，因为使拉格朗日函数最大的那个 &alpha;<sup>'</sup> 也包含在任意一个中，所以我们也可以把 &alpha;<sup>'</sup> 看做是 &alpha;（**拉格朗日对偶问题**）：

<img src="http://latex.codecogs.com/svg.latex?{\min_{b,\mathbf{w}}\left(\max_\mathrm{{all}\,\alpha_n\,\ge\,0}\,\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})\right)\,\ge\,\underbrace{\max_\mathrm{{all}\,\alpha_n\,\ge\,0}\left(\min_{b,\mathbf{w}}\,\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})\right)}_{\textrm{Lagrange\;dual\;problem}}}"/>

这个拉格朗日对偶问题是 SVM 问题的下限（小于等于），下限是一个弱对偶关系。如果等号成立的话（强对偶关系），我们就可以用右边的问题代替原来的 SVM，这样的好处是右边对 b 和 **w** 的最小化是没有条件的，很好解。

如果这个问题是凸的、有解（线性可分）并且只有线性的限制条件，这个`强对偶关系`就成立，SVM 符合这些条件，等号成立（这里“文科生”了）。

---

下面来求解（化简）拉格朗日对偶问题：

<img src="http://latex.codecogs.com/svg.latex?{\max_\mathrm{{all}\,\alpha_n\,\ge\,0}\left(\min_{b,\mathbf{w}}\underbrace{\frac{1}{2}\mathbf{w}^T\mathbf{w}\;+\;\sum_{n=1}^N\alpha_n(1-\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b)}_{\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})}\right)}"/>

因为`拉格朗日函数`的最小化没有条件，所以只要求导（偏微分导数）就可以了，拉格朗日函数是 b 、 **w** 和 &alpha; 的函数，我们先对 b 求偏微分：

<img src="http://latex.codecogs.com/svg.latex?{\frac{\partial\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})}{\partial{b}}=-\sum^{N}_{n=1}\alpha_n\mathrm{y}_n=0}"/>

既然最佳解满足这个偏微分导数为 0 的条件，我们可以把这个条件加进拉格朗日对偶问题中：

<img src="http://latex.codecogs.com/svg.latex?{\max_\mathrm{{all}\,\alpha_n\,\ge\,0,\sum\alpha_n\mathrm{y}_n=0}\left(\min_{b,\mathbf{w}}{\frac{1}{2}\mathbf{w}^T\mathbf{w}\;+\;\sum_{n=1}^N\alpha_n(1-\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n)}\right)}"/>

同时，因为 b 的系数为 0 因此 b 被去掉了。但是 b 在我们做预测的时候是有用的（需要计算），没关系，后面有办法再把 b 算出来。

那么我们继续对 **w** 求偏微分：

<img src="http://latex.codecogs.com/svg.latex?{\frac{\partial\mathcal{L}(b,\mathbf{w},\boldsymbol{\alpha})}{\partial\mathrm{w}_{i}}=\mathrm{w}_{i}-\sum^{N}_{n=1}\alpha_n\mathrm{y}_n\mathrm{z}_{n,i}=0}"/>

向量化表示：

<img src="http://latex.codecogs.com/svg.latex?{\mathbf{w}=\sum^{N}_{n=1}\alpha_n\mathrm{y}_n\mathbf{z}_n}"/>

继续带入：

<img src="http://latex.codecogs.com/svg.latex?{\max_\mathrm{{all}\,\alpha_n\,\ge\,0,\sum\alpha_n\mathrm{y}_n=0,\mathbf{w}=\sum\alpha_n\mathrm{y}_n\mathbf{z}_n}\left(-\frac{1}{2}\|\sum^{N}_{n=1}\alpha_n\mathrm{y}_n\mathbf{z}_n\|^2+\sum^{N}_{n=1}\alpha_n\right)}"/>

现在这个问题就只是和 &alpha; 相关的最佳化问题，总结一下现在所有的条件（ **KKT 条件**）：

![](./Snapshot/Snap06.png)

### 求解对偶 SVM

现在 SVM 问题被转化成了只和 &alpha; 相关的最佳化问题，我们再来做一些简化。首先我们通过取负把最大化问题转化成最小化问题：（因为我们现在只讨论 &alpha; 所以和 **w** 相关的条件可以不用处理）

<img src="http://latex.codecogs.com/svg.latex?{\min_{\boldsymbol{\alpha}}\frac{1}{2}\sum^{N}_{n=1}\sum^{N}_{m=1}\alpha_n\alpha_m\mathrm{y}_n\mathrm{y}_m\mathbf{z}_n^T\mathbf{z}_m-\sum^{N}_{n=1}\alpha_n\textrm{\,\,s.\,t.\,\,}\sum_{n=1}^{N}\alpha_n\mathrm{y}_n=0;\;\alpha_n\,\ge\,0,\textrm{for}\,n=1,\dots,N}"/>

这个最小化问题有 N 个变量（&alpha;）和 N+1 个条件！并且这个问题也是一个凸的二次规划问题！下面只要把这个问题转化成标准二次规划问题的形式就好了！

![](./Snapshot/Snap07.png)

但是转化后的二次规划问题也不好解，因为这里有个巨大的非零矩阵 Q ，而且这个 Q 还需要先算出来，再放到求解二次规划的工具中，通常需要花费太长的计算时间。

同时，其实我们也没有真正的摆脱 <i>d</i>，因为当 <i>d</i> 很大的时候，计算 Q 仍然和 <i>d</i> 是有关的，这个我们在下一讲来处理。

实际上，通常有特别为 SVM 求解的二次规划工具，专门针对求解 SVM 做了优化，加速计算 Q 矩阵的过程。

---

当我们得到了 &alpha; 之后，我们需要用 KKT 条件来算出 b 和 **w**：

<img src="http://latex.codecogs.com/svg.latex?{\mathbf{w}=\sum_{n=1}^N\alpha_n\mathrm{y}_n\mathbf{z}_n}"/>

对于 b ，如果 &alpha;<sub>n</sub> > 0 ，则可利用拉格朗日函数中的条件项也可以算出：

<img src="http://latex.codecogs.com/svg.latex?{\alpha_n(1-\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b))=0\;\Rightarrow\;b=\mathrm{y}_n-\mathbf{w}^T\mathbf{z}_n}"/>

这些 &alpha;<sub>n</sub> > 0 的那些数据点就是在边界上的`支持向量`。

在边界上的那些点是`支持向量`的候选，而那些 &alpha;<sub>n</sub> > 0 的点才是真的`支持向量`。

### SVM 其他的信息

既然那些 &alpha;<sub>n</sub> > 0 的点才是真的支持向量，那么在计算 **w** 的时候，只需要计算支持向量的那些项：

<img src="http://latex.codecogs.com/svg.latex?{\mathbf{w}=\sum_{n=1}^N\alpha_n\mathrm{y}_n\mathbf{z}_n=\sum_\textrm{SV}\alpha_n\mathrm{y}_n\mathbf{z}_n}"/>

而 b 则可以用任意一个支持向量计算出来。

这和之前我们讲 SVM 的几何意义是相似的，只有支持向量有用，而其他的数据没有用。因此 SVM 也可以看成是一个找出支持向量的机制！

类似的，PLA 则是用“犯错误”的数据计算 **w** 算法：

<img src="http://latex.codecogs.com/svg.latex?{\mathbf{w}_{\textrm{SVM}}=\sum_{n=1}^N\alpha_n\mathrm{y}_n\mathbf{z}_n;\quad\mathbf{w}_{\textrm{PLA}}=\sum_{n=1}^N\beta_n\mathrm{y}_n\mathbf{z}_n;}"/>

这就很有意思了，这些算法都是数据的`线性组合`，也就说 **w** 可以用数据`表现` (represent) 出来。

---
---
---

## Lecture 3: Kernel Support Vector Machine

—— 介绍核函数的概念

—— 介绍不同的核函数和其特点

### 核函数

在我们利用对偶问题和二次规划求解 SVM 的时候，还剩下一个问题就是 Q 矩阵的计算是依赖于 <i>d</i> 的：

<img src="http://latex.codecogs.com/svg.latex?{q_{n,m}=\mathrm{y}_n\mathrm{y}_m\mathbf{z}^T_n\mathbf{z}_m;\quad\mathbf{z}^T_n\mathbf{z}_m=\boldsymbol{\Phi}(\mathbf{x}_n)^T\boldsymbol{\Phi}(\mathbf{x}_m)\in\mathbb{R}^{\tilde{d}}\,-\,O(\tilde{d})}"/>

这里面最关键的计算就是是先做`特征转换`再做`內积`，导致计算很复杂。那如果我们可以先做`內积`再做`特征转换` **也许** 就可以避开复杂的计算。

先用一个二项式的特征转换来看一下：

![](./Snapshot/Snap08.png)

果然，在这个例子上可以减少计算复杂度。

我们把这个`特征转换`和`內积`的函数叫做 **核函数 (kernal function)**。

这样一来，刚才说的 Q 矩阵可以用核函数来节约计算：

<img src="http://latex.codecogs.com/svg.latex?{q_{n,m}=\mathrm{y}_n\mathrm{y}_m\mathbf{z}^T_n\mathbf{z}_m=\mathrm{y}_n\mathrm{y}_{m}K(\mathbf{x}_n,\mathbf{x}_m)}"/>

不仅如此，还有很多东西都可以用核函数来节约计算，比如 b：

<img src="http://latex.codecogs.com/svg.latex?{b=\mathrm{y}_\textrm{sv}-\mathbf{w}^T\mathbf{z}_\textrm{sv}}=\mathrm{y}_\textrm{sv}-\left(\sum_{n=1}^N\alpha_n\mathrm{y}_n\mathbf{z}_n\right)\mathbf{z}_\textrm{sv}=\mathrm{y}_\textrm{sv}-\sum_{n=1}^N\alpha_n\mathrm{y}_{n}\left\Big(K(\mathbf{x}_n,\mathbf{x}_\textrm{sv})\right\Big)}"/>

还有做预测的时候：

<img src="http://latex.codecogs.com/svg.latex?{g_\textrm{svm}(\mathbf{x})=\mathrm{sign}(\mathbf{w}^T\boldsymbol{\Phi}(\mathbf{x})+b)=\mathrm{sign}\left(\sum_{n=1}^N\alpha_n\mathrm{y}_{n}K(\mathbf{x}_n,\mathbf{x})+b\right)}"/>

通过`核函数`，我们其实没有任何一次是在特征转换后的 **z** 空间做运算，这样我们也就真正意义上的摆脱了特征转换的复杂度 <i>d</i> ，只需要 **x** 就可以！这就是 **Kernel SVM** ！

---

### 多项式核函数

刚才我们看了一下二项式转换和相对应的核函数，如果我们对二次项做一个缩放，一次项再做一个根号二倍的缩放，就可以得到一个简单的、一般形式的 **二项式核函数** 的表现形式。

<img src="http://latex.codecogs.com/svg.latex?{K_2(\mathbf{x},\mathbf{x}')=(1+\gamma\mathbf{x}^T\mathbf{x}')^2\;\textrm{with}\;\gamma>0}"/>

这个一般形式的核函数和原本的二项式核函数有什么区别呢？

它们对应到同一个二次空间，因为都是二次的特征转换；但是因为系数不同，尤其是在 **x** 的`內积`
上，在`几何`上定义了不同的`距离`，因此会得到不同的`边界`：

![](./Snapshot/Snap09.png)

把二项式核函数进一步扩展下去，就可以得到 **多项式核函数** 的一般形式：

<img src="http://latex.codecogs.com/svg.latex?{K_Q(\mathbf{x},\mathbf{x}')=(\zeta+\gamma\mathbf{x}^T\mathbf{x}')^Q\;\textrm{with}\;\gamma>0,\zeta\,\ge\,0}"/>

利用核函数，我们可以很方便的使用很高次的特征转换而不用花费太多的计算；同时利用边界，可以控制函数的复杂度，帮助我们不会过拟合。

当然，还有最简单的 **线性核函数** （略）。

### 高斯核函数

既然我们可以用核函数摆脱特征转换 <i>d</i> 的限制，也节省了很多计算，那么我们是不是可以疯狂的把特征转换扩展，甚至扩展到 **无限多** 的维度上？

为了简化这个问题，我们现在假设输入 **x** 只有一维。然后我们构建一个简单的高斯函数，并来证明这个高斯函数相当于是在无限多维上的特征转换！

<img src="http://latex.codecogs.com/svg.latex?{\begin{align*}K(\mathrm{x},\mathrm{x}')&\quad=\quad\exp(-(\mathrm{x}-\mathrm{x}')^2)\\&\quad=\quad\exp(-(\mathrm{x})^2)\exp(-(\mathrm{x}')^2)\exp(2\mathrm{x}\mathrm{x}')\\&\;\,\overset{\textrm{Taylor}}{=}\,\;\exp(-(\mathrm{x})^2)\exp(-(\mathrm{x}')^2)\left(\sum_{i=0}^{\infty}\frac{(2\mathrm{x}\mathrm{x}')^i}{i!}\right)\\&\quad=\quad\sum_{i=0}^{\infty}\left(\sqrt{\frac{2^i}{i!}}\exp(-(\mathrm{x})^2)(\mathrm{x})^i\sqrt{\frac{2^i}{i!}}\exp(-(\mathrm{x}')^2)(\mathrm{x}')^i\right)\\&\quad=\quad\boldsymbol{\Phi}(\mathrm{x})^T\boldsymbol{\Phi}(\mathrm{x}')\end{align*}}"/>

这里的：

<img src="http://latex.codecogs.com/svg.latex?{\boldsymbol{\Phi}(\mathrm{x})=\exp(-\mathrm{x}^2)\cdot\,\left(1,\sqrt{\frac{2}{1!}}\mathrm{x},\sqrt{\frac{2^2}{2!}}\mathrm{x}^2,\dots\right)}"/>

通过泰勒展开，高斯函数里可以藏着一个无限多维特征转换的核函数 **高斯核函数**！是不是很神奇！

更一般化一点的：

<img src="http://latex.codecogs.com/svg.latex?{K(\mathbf{x},\mathbf{x}')=\exp(-\gamma\|\mathbf{x}-\mathbf{x}')\|^2)\;\textrm{with}\;\gamma>0}"/>

当我们预测的时候：

<img src="http://latex.codecogs.com/svg.latex?{g_\textrm{svm}(\mathbf{x})=\mathrm{sign}\left(\sum_{\texnrm{SV}}\alpha_n\mathrm{y}_{n}\exp(-\gamma\|\mathbf{x}-\mathbf{x}_n)\|^2)+b\right)}"/>

这相当于是很多以`支持向量`为中心的`高斯函数`进行`线性组合`，因此这个核函数又叫 **径向基核函数 (Radial Basis Function, RBF)**。

不过，尽管 SVM 有`核函数`帮助我们省掉了很多计算复杂度，有`边界`帮助我们控制复杂度，但在这个强大的`特征转换`下，仍然有可能会`过拟合`！所以一定要慎重选择参数。

![](./Snapshot/Snap10.png)

### 核函数的比较

下面我们来总结比较一下这几个核函数的优点和缺点：

- 线性核函数
    - 简单，快速（不需要转换成对偶问题），可解释的
    - 存在线性可分的限制
- 多项式核函数
    - 比线性更强，可以根据对问题的理解控制多项式系数（通常比较小的 Q ）
    - 计算比较困难（数值范围比较大），需要调整的参数多
- 高斯/径向基核函数
    - 超级强大，容易计算，参数简单
    - 过拟合风险，模型难以解释

其实还有其他的核函数，核函数的几何意义是两个向量的內积（相似性），那么是不是任意某个`相似性`的表示都可以作为核函数呢？不是的。因为核函数还有一些特性的要求（充分必要条件）：
- 对称性
- K 矩阵是半正定的

不过“创造”一个新的核函数还是很难的。

---
---
---

## Lecture 4: Soft-Margin Support Vector Machine

—— 介绍软边界的支持向量机（也就是一般所说的SVM）

### 软边界的意义

上面我们介绍了硬边界的 SVM，它要求数据必须是完全正确区分的。然而实际上这个条件会引发很多问题，比如数据有噪音，追求完全正确区分不一定有意义；而且能够完全正确区分意味着可以`Shatter`，复杂度会偏高；这些在实际应用中反而会容易造成过拟合。

因此，实际应用中我们应该能够容忍一些噪音。在《基石》中，我们讲过一个`口袋算法`，使分类器在数据集中出现的错误最少，我们也可以把它运用在 SVM 上：

<img src="http://latex.codecogs.com/svg.latex?{\begin{align*}\,\min_{b,\mathbf{w}}\,&\,\frac{1}{2}\mathbf{w}^T\mathbf{w}+C\cdot\sum_{n=1}^{N}[\![\mathrm{y}_n\ne\textrm{sign}(\mathbf{w}^T\mathbf{z}_n+b)]\!]\\\textrm{\,\,s.\,t.\,\,}&\,\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b)\,\ge\,1-\infty\cdot[\![\mathrm{y}_n\ne\textrm{sign}(\mathbf{w}^T\mathbf{z}_n+b)]\!]\end{align*}}"/>

这里 的 C 是可以用来调节（权衡）`边界`和`噪音容忍程度`的参数。

但是这样一来这个问题就不是一个 QP 问题了，因此之前我们做的所有努力就都不能用了。除此以外，小和大的错误没有区分，显然这也是不好的。

因此我们提出一个新的参数：“边界违背程度”，用这个参数来代替错误的个数来表示错误，这样就可以保证我们仍然是一个 QP 问题，这就是软边界的 SVM：

<img src="http://latex.codecogs.com/svg.latex?{\begin{align*}\,\min_{b,\mathbf{w},\boldsymbol{\xi}}\,&\,\frac{1}{2}\mathbf{w}^T\mathbf{w}+C\cdot\sum_{n=1}^{N}\xi_n\\\textrm{\,\,s.\,t.\,\,}&\,\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b)\,\ge\,1-\xi_n\;\textrm{and}\;\xi_n\,\ge\,0\end{align*}}"/>

这是一个有 <i>d</i>+1+N 个变量和 2N 个条件 的 QP，我们接下来就可以用和前面硬边界 SVM 类似的方法来解决它。

### 对偶问题

既然现在我们得到的问题和，硬边界 SVM 是类似的 QP 问题，那么解决的方法也是类似的，我们还是先构建一个拉格朗日函数来化解这个问题：

<img src="http://latex.codecogs.com/svg.latex?{\mathcal{L}(b,\mathbf{w},\boldsymbol{\xi},\boldsymbol{\alpha},\boldsymbol{\beta})={\frac{1}{2}\mathbf{w}^T\mathbf{w}}+C\cdot\sum_{n=1}^{N}\xi_n\;+\;\sum_{n=1}^N\alpha_n\cdot(1-\xi_n-\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b))+\sum_{n=1}^N\beta_n\cdot(-\xi_n)}"/>

拉格朗日对偶问题：

<img src="http://latex.codecogs.com/svg.latex?{\max_{\alpha_n\,\ge\,0,\beta_n\,\ge\,0}\left(\min_{b,\mathbf{w},\boldsymbol{\xi}}\,\mathcal{L}(b,\mathbf{w},\boldsymbol{\xi},\boldsymbol{\alpha},\boldsymbol{\beta})\right)}"/>

求导：

<img src="http://latex.codecogs.com/svg.latex?{\frac{\partial\mathcal{L}}{\partial{\xi_n}}=C-\alpha_n-\beta_n=0}"/>

这个新的条件就可以帮助我们简化，带入之后神奇的消掉了 &xi;<sub>n</sub>（和之前消掉 b 类似），于是得到：

<img src="http://latex.codecogs.com/svg.latex?{\max_{0\,\le\,\alpha_n\,\le\,C}\left(\min_{b,\mathbf{w}}{\frac{1}{2}\mathbf{w}^T\mathbf{w}\;+\;\sum_{n=1}^N\alpha_n(1-\mathrm{y}_n(\mathbf{w}^T\mathbf{z}_n+b)}\right)}"/>

除了 &alpha;<sub>n</sub> 多了一个上限是 C 的条件以外，其他和之前的 SVM 一样！因此之前的所有努力都可以继续用到这个新的 SVM 上来。

### 其他的信息

软边界 SVM 和硬边界 SVM 只有一点点区别 （上限是 C），利用 KKT 条件计算的时候有一点区别：

![](./Snapshot/Snap11.png)

这种在上限 C 以下的支持向量（SV）叫做 free SV。这样一来软边界的 SVM 把数据分成了三类：
1. non-SV (&alpha; = 0, &xi;<sub>n</sub> = 0)，距离边界很远的点，对找到超平面没有影响；
2. free-SV (0 < &alpha;<sub>n</sub> < C, &xi;<sub>n</sub> = 0)，在边界上的点；
3. 边界内的点 (&alpha;<sub>n</sub> = C, &xi;<sub>n</sub> = 边界违背程度)；

利用这个数据分类，可以帮助我们分析数据，到底哪些是有问题的分类，哪些是噪音等。

不过需要注意的是，当 C 越大，软边界的 SVM 和硬边界的 SVM 就越像（边界上的容忍很低），因此也是有可能过拟合的！

### 模型选择



<!--  -->
