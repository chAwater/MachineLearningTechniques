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
- <img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\mathrm{sign}(\mathbf{w}^T\mathbf{x})"/>
- PLA 算法

对于一些问题，可能存在多种 PLA 的解都可以把数据分开，得到这些解取决于把数据放入 PLA 算法的顺序，并且看上去“一样好”。

![](./Snapshot/Snap01.png)

但是我们 **人类** 会倾向于选择最右边的这个，这是因为当存在一些噪声的时候（比如测试数据和训练数据之间存在一些误差）最右边的这个线可以容忍最多的噪声、误差。

所以我们希望 **每个点都和我们的线距离最远** ，也可以说我们希望我们能够找到一个 **最胖** 的线，这个线离它最近的点的距离最远。
这个线有多“胖”，就是说这个线的边界有多大，称为 **边界 (margin)**。

总结一下：
- 找到一个可以正确区分数据的线性分类器
- 得到每个数据和这个线性分类器的距离，取最小的距离作为边界
- 最大化这个边界

![](./Snapshot/Snap02.png)

---




<!--  -->
