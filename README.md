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

但是我们 **人类** 会倾向于选择最右边的这个，这是因为当存在一些噪声的时候（比如测试数据和训练数据之间存在一些误差）最右边的这个线可以容忍最多的噪声、误差。

所以我们希望 **每个点都和我们的线距离最远**，也可以说我们希望我们能够找到一个 **最胖** 的线，这个线离它最近的点的距离最远。
这个线有多“胖”，就是说这个线的`边界`( margin ) 有多大。

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

[Issues #1]

---

#### 简化距离

对于每个数据点和分类器（超平面）之间的距离：
- 考虑超平面上的任意一点 x<sup>'</sup>
- **w** 与超平面上任意一点的乘积为 0 ，因此 **w** 相当于超平面的法向量（<img src="http://latex.codecogs.com/svg.latex?{\mathbf{w}^T\mathbf{x}^{'}+b=0}"/>）
<!-- （![](https://render.githubusercontent.com/render/math?math=\mathbf{w}^T\mathbf{x}^{'}%2Bb=0)） -->

- 数据点和超平面的距离，相当于数据点和连接的向量在垂直于超平面方向（ w ）上的投影
- 因为这数据点可以被分类器（超平面）区分，因此有 <img src="http://latex.codecogs.com/svg.latex?{\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)>0}"/>
<!-- ![](https://render.githubusercontent.com/render/math?math=\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)\gt0) -->

所以距离可以简化为：

<img src="http://latex.codecogs.com/svg.latex?{\mathrm{distance}(\mathbf{x},b,\mathbf{w})=|\frac{\mathbf{w}^T}{||\mathbf{w}||}(\mathbf{x}-\mathbf{x}')|=\frac{1}{||\mathbf{w}||}|\mathbf{w}^T+b|=\frac{1}{||\mathbf{w}||}\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)}"/>

<!-- ![](https://render.githubusercontent.com/render/math?math=\mathrm{distance}\(\mathbf{x},b,\mathbf{w}\)=\left|\frac{\mathbf{w}^T}{||\mathbf{w}||}\(\mathbf{x}-\mathbf{x}^{'}\)\right|=\frac{1}{||\mathbf{w}||}|\mathbf{w}^T%2Bb|=\frac{1}{||\mathbf{w}||}\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)) -->

#### 简化条件 - 向量缩放和边界的定义

对于表示这个分类器（超平面）的向量来说，向量的缩放（改变长度不改变方向）并不影响，仍然可以表示这个超平面。

因此，我们可以对这个向量进行一个 **特殊的缩放**，使：
<img src="http://latex.codecogs.com/svg.latex?{\mathop{\min_{n=1,...,N}}\,\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)=1}"/>

<!-- ![](https://render.githubusercontent.com/render/math?math=\underset{n=1\,\\!\\!...\,\\!\\!N}{\min}\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)=1) -->

这样操作之后有两个好处：
1. 显然在这种缩放下可以保证 <img src="http://latex.codecogs.com/svg.latex?{\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)>0}"/>，因此这个条件可以去掉；
2. 边界变成 <img src="http://latex.codecogs.com/svg.latex?{\frac{1}{||\mathbf{w}||}}"/>

所以这个问题就被简化为：

<img src="http://latex.codecogs.com/svg.latex?{\mathop{\max_{b,\mathbf{w}}}\frac{1}{||\mathbf{w}||}\mathrm{\quad\,subject\,to\quad\,}\,\mathop{\min_{n=1,...,N}}\,\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)=1}"/>

#### 简化条件 - 有帮助的宽松

我们继续简化这个问题，我们先将条件 **放宽** 到：对于所有的数据（所有的 n ）有 <img src="http://latex.codecogs.com/svg.latex?{\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)\,\ge\,1}"/>

如果 **大于等于** 中的 **等于** 对于宽松后的解不成立，那么这个 **放宽** 后的问题与之前的问题是不同的。

但是，如果等于不成立，那么我们必然可以找到一个新的 **缩放** 使得 **等于** 成立，并且这个新的 **缩放** 是比原来的 **缩放** 程度更大，因此 || **w** || 只会更小，因此我们能找到一个更大的 **边界**，这就与宽松后的解产生了矛盾，因此这个宽松中的等于是必然成立的。

然后我们通过取倒数将最大化问题转成最小化问题，并用平方去掉绝对值（根号），再加上一个常数项。

![](./Snapshot/Snap03.png)

### 支持向量机

这种算法被称为`支持向量机`(Support Vector Machine, SVM)，是因为在超平面`边界`上的那些数据点决定了这个超平面和边界，而其他地方的数据点对于边界和超平面来说是不必要的。
这些在超平面边界上的点被称为`支持向量`（的候选），因为这些点就好像在支撑着这个超平面一样。

那么我们继续来求解这个问题，这个问题有一些特性：
- 这个问题是`凸的二次函数`
- 这个问题是 **w** 和 b 的`线性运算`

具有这种特性的问题被称为`二次规划`( Quadratic Programming, QP )，有很多现成的工具来求解这种问题，那么我们只要把这个问题转化成标准二次规划问题的形式就很好处理了。

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


<!--  -->
