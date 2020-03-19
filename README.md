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

- 需要先学习《机器学习基石》再学习《机器学习技法》

### 其他支持

- [<img class="emoji" title=":atom:" alt=":atom:" src="https://github.githubassets.com/images/icons/emoji/atom.png" height="20" width="20" align="absmiddle"> Atom](https://atom.io)
- [CodeCogs (LaTeX Editor API)](http://latex.codecogs.com)
- [Grip -- GitHub Readme Instant Preview](https://github.com/joeyespo/grip)
- [Markdown Toc](https://github.com/nok/markdown-toc)

---

## Lecture 1: Linear Support Vector Machine

—— 介绍边界的概念

### 最大边界的线性分类器

先回顾一下线性分类问题：
<!-- - <img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\mathrm{sign}(\mathbf{w}^T\mathbf{x})"/> -->
- ![](https://render.githubusercontent.com/render/math?math=h\(\mathbf{x}\)=\mathrm{sign}\(\mathbf{w}^T\mathbf{x}\))
- PLA 算法

对于一些问题，可能存在多种 PLA 的解都可以把数据分开，得到这些解取决于把数据放入 PLA 算法的顺序，并且看上去“一样好”。

![](./Snapshot/Snap01.png)

但是我们 **人类** 会倾向于选择最右边的这个，这是因为当存在一些噪声的时候（比如测试数据和训练数据之间存在一些误差）最右边的这个线可以容忍最多的噪声、误差。

所以我们希望 **每个点都和我们的线距离最远** ，也可以说我们希望我们能够找到一个 **最胖** 的线，这个线离它最近的点的距离最远。
这个线有多“胖”，就是说这个线的边界有多大，称为 **边界 (margin)**。

总结一下：
- 找到一个可以正确区分数据的线性分类器（超平面）
- 得到每个数据和这个线性分类器的距离，取最小的距离作为边界
- 最大化这个边界

![](./Snapshot/Snap02.png)

---

### 简化问题

首先，需要把线性分类问题中的 w<sub>0</sub> 单独拿出来讨论，称之为 b，并把 x<sub>0</sub> 从原先的 x 中去掉：

<!-- <img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\mathrm{sign}(\mathbf{w}^T\mathbf{x}+b)"/> -->
![](https://render.githubusercontent.com/render/math?math=h\(\mathbf{x}\)=\mathrm{sign}\(\mathbf{w}^T\mathbf{x}%2Bb\))

对于每个数据点和分类器（超平面）之间的距离：
- 考虑超平面上的任意一点 x<sup>'</sup>
- w 与超平面上任意一点的乘积为 0 ，因此 w 相当于超平面的法向量 （![](https://render.githubusercontent.com/render/math?math=\mathbf{w}^T\mathbf{x}^{'}%2Bb=0)）
<!-- （<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}^T\mathbf{x'}+b=0"/>） -->

- 数据点和超平面的距离，相当于数据点和连接的向量在垂直于超平面方向（ w ）上的投影
- 因为这数据点可以被分类器（超平面）区分，因此有 ![](https://render.githubusercontent.com/render/math?math=\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)\gt0)
<!-- <img src="http://latex.codecogs.com/svg.latex?\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)>0"/> -->

所以距离可以简化为：

<!-- <img src="http://latex.codecogs.com/svg.latex?\mathrm{distance}(\mathbf{x},b,\mathbf{w}))=|\frac{\mathbf{w}^T}{||\mathbf{w}||}(\mathbf{x}-\mathbf{x}')|=\frac{1}{||\mathbf{w}||}|\mathbf{w}^T+b|=\frac{1}{||\mathbf{w}||}\mathrm{y}_n(\mathbf{w}^T\mathbf{x}_n+b)"/> -->
![](https://render.githubusercontent.com/render/math?math=\mathrm{distance}\(\mathbf{x},b,\mathbf{w}\)=\left|\frac{\mathbf{w}^T}{||\mathbf{w}||}\(\mathbf{x}-\mathbf{x}^{'}\)\right|=\frac{1}{||\mathbf{w}||}|\mathbf{w}^T%2Bb|=\frac{1}{||\mathbf{w}||}\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\))

---

对于表示这个分类器（超平面）的向量来说，向量的缩放（改变长度不改变方向）并不影响，仍然可以表示这个超平面。

因此，我们可以对这个向量进行一个特殊的缩放，使 ![](https://render.githubusercontent.com/render/math?math=\underset{n=1\,\\!\\!...\,\\!\\!N}{\min}\mathrm{y}_n\(\mathbf{w}^T\mathbf{x}_n%2Bb\)=1)

---

[Issues #1]

---

<!--  -->
