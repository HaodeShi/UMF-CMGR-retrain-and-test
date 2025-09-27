#  说明


# 本代码在开源工作 https://github.com/wdhudiekou/UMF-CMGR  上进行训练和测试 ，如有侵权，请联系本人删除 Q：1843945104

# 非常感谢原作者 Di Wang, Jinyuan Liu, Xin Fan, and Risheng Liu  的开源工作



# UMFusion（无监督错位红外与可见光图像融合模型）


[![许可证](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python版本](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch版本](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)


### 基于跨模态图像生成与配准的无监督错位红外与可见光图像融合【国际人工智能联合会议（IJCAI）2022年口头报告】

作者：王迪、刘金源、范鑫、刘日升

<div align=center>
<img src="https://github.com/wdhudiekou/UMF-CMGR/blob/main/Fig/network.png" width="80%">
</div>


## 更新记录
[2022-07-14] 配准网络（MRRN）与融合网络（DIFN）的预训练模型已发布！  
[2022-06-21] 跨模态生成模型（CPSTN）已发布！  
[2022-05-30] 论文中文翻译版本已上线，欢迎查阅！[[中译版本](./pdf/CN.pdf)]  
[2022-05-25] 论文在线版已发布！[[arXiv版本](https://arxiv.org/pdf/2205.11876.pdf)]  


## 环境依赖
- CUDA 10.1（显卡计算架构，保留原名）
- Python 3.6（或更高版本）
- PyTorch 1.6.0（深度学习框架，保留原名）
- Torchvision 0.7.0（PyTorch配套视觉工具库，保留原名）
- OpenCV 3.4（计算机视觉库，保留原名）
- Kornia 0.5.11（PyTorch视觉算法库，保留原名）






# 第一步生成伪红外图像

# 第二步构造形变红外图像

# 第三步配准伪红外与变形红外

# 第四步融合
 





# BUG 
## XXX 数据准备
1. 可通过以下命令获取训练/测试所需的变形红外图像：
    ```python
       cd ./data  # 进入data文件夹
       python get_test_data.py  # 运行数据获取脚本
    ```
   在`Trainer/train_reg.py`（配准训练脚本）中，默认在训练过程中实时生成可变形红外图像。

2. 可通过以下命令获取红外-可见光图像融合（IVIF）训练所需的自视觉显著性图：
    ```python
       cd ./data  # 进入data文件夹
       python get_svs_map.py  # 运行显著性图生成脚本
    ```


## XXX 快速开始
1. 可使用我们的CPSTN模型生成的伪红外图像[[链接](https://pan.baidu.com/s/1M79RuHVe6udKhcJIA7yXgA) 提取码：qqyj]，进行配准过程的训练/测试：
    ```python
       cd ./Trainer  # 进入训练文件夹
       python train_reg.py  # 训练配准网络

       cd ./Test  # 进入测试文件夹
       python test_reg.py  # 测试配准网络
    ```
   请下载配准网络（MRRN）的[预训练模型](https://pan.baidu.com/s/199dqOLHyJS9aY5YecuVglA)（提取码：hk25）。

2. XXX 若需使用CPSTN模型为其他数据集生成伪红外图像，可直接运行以下命令：
    ```python
    ## 测试阶段
       cd ./CPSTN  # 进入CPSTN模型文件夹
       python test.py --dataroot datasets/rgb2ir/RoadScene/testA --name rgb2ir_paired_Road_edge_pretrained --model test --no_dropout --preprocess none
    
    ## 训练阶段
       cd ./CPSTN  # 进入CPSTN模型文件夹
       python train.py --dataroot ./datasets/rgb2ir/RoadScene --name rgb2ir_paired_Road_edge --model cycle_gan --dataset_mode unaligned
    ```
   CPSTN模型的训练与测试数据可从以下链接下载：[数据集](https://pan.baidu.com/s/1-U1n945ykHFU7yrEHwGC9Q)（提取码：u386）。  
   请下载CPSTN的[预训练模型](https://pan.baidu.com/s/1JO4hjdaXPUScCI6oFtPEnQ)（提取码：i9ju），并将其放入文件夹`./CPSTN/checkpoints/pretrained/`（CPSTN预训练模型检查点文件夹）中。

3. XXX 若需分别训练配准与融合过程，可运行以下命令：
    ```python
       cd ./Trainer  # 进入训练文件夹
       python train_reg.py  # 训练配准网络

       cd ./Trainer  # 进入训练文件夹
       python train_fuse.py  # 训练融合网络
    ```
   对应的测试脚本`test_reg.py`（配准测试）和`test_fuse.py`（融合测试）可在`Test`（测试文件夹）中找到。请下载融合网络（DIFN）的[预训练模型](https://pan.baidu.com/s/1GZrYrg_qzAfQtoCrZLJsSw)（提取码：0rbm）。

4. XXX 若需联合训练配准与融合过程，可运行以下命令：
   ```python
       cd ./Trainer  # 进入训练文件夹
       python train_reg_fusion.py  # 联合训练配准与融合网络
   ```
   对应的测试脚本`test_reg_fusion.py`（联合测试）可在`Test`（测试文件夹）中找到。





## 数据集
请下载以下数据集：
*   [RoadScene](https://github.com/hanna-xu/RoadScene)（道路场景图像数据集，保留原名）
*   [TNO](http://figshare.com/articles/TNO\_Image\_Fusion\_Dataset/1008029)（TNO图像融合数据集，保留原名）


## BUG 实验结果
请下载以下由CPSTN模型生成的伪红外图像：
*  [伪红外图像](https://pan.baidu.com/s/1M79RuHVe6udKhcJIA7yXgA)（提取码：qqyj）

请下载以下由UMF模型得到的配准后红外图像：
*  [RoadScene数据集配准结果](https://pan.baidu.com/s/161lbmGx8TDphx0Uf9cAtfg )（提取码：4cx2）
*  [TNO数据集配准结果](https://pan.baidu.com/s/1AO2T4LMsujIQcrJT9WnHpg )（提取码：2edi）

请下载以下由UMF模型得到的融合图像：
*  [RoadScene数据集融合结果](https://pan.baidu.com/s/1aG_CI9fFIhsV2Z2ThMUPQg )（提取码：1zuu）
*  [TNO数据集融合结果](https://pan.baidu.com/s/10Me7GpM_tvHgkWVzv2pv3g )（提取码：22gc）

<div align=center>
<img src="https://github.com/wdhudiekou/UMF-CMGR/blob/main/Fig/quntitative_results.png" width="95%">
</div>

<div align=center>
<img src="https://github.com/wdhudiekou/UMF-CMGR/blob/main/Fig/qualitative_results.png" width="95%">
</div>


## 相关项目

*  [IMF](https://github.com/wdhudiekou/IMF)（UMF的改进版本，已被《IEEE Transactions on Circuits and Systems for Video Technology》（IEEE TCSVT）2024年刊收录）


## 引用格式
```
@inproceedings{UMF,
	author    = {Di Wang and
	Jinyuan Liu and
	Xin Fan and
	Risheng Liu},
	title     = {Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration},
	booktitle = {IJCAI},  # 国际人工智能联合会议（IJCAI）
	pages     = {3508--3515},
	year      = {2022}
}
```




#   python test.py --dataroot D:/3_SCI_3/SCI_3_code/02_UMF-CMGR-main/dataset/Roadscene/vi_256 --name rgb2ir_paired_Road_edge_pretrained --model test --no_dropout --preprocess none


D:/3_SCI_3/SCI_3_code/02_UMF-CMGR-main/dataset/Roadscene/vi_256