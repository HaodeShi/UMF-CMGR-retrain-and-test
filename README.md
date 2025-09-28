## 1、项目简介


#### 本项目在开源工作 UMF-CMGR 上进行训练和测试 ，如有侵权，请联系本人删除 Q：1843945104

#### 本项目数据集和训练权重链接: https://pan.baidu.com/s/1exPI31jwuy7suS_ZwnzXjw 提取码: 1234

#### 非常感谢原作者 Di Wang, Jinyuan Liu, Xin Fan, and Risheng Liu  的开源工作

#### 论文题目《Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration》

#### 原项目开源地址：https://github.com/wdhudiekou/UMF-CMGR

#### ！！！！大部分问题都可以在原项目Issues中找到

## 2、本项目主要工作（重新进行了训练和测试）

   1. 开源项目测试
   ```python

      Stage 1: 下载开源代码 https://github.com/wdhudiekou/UMF-CMGR

      Stage 2: 下载测试数据集（需要自己准备），本项目数据在dataset\Roadscene中，具体结构为
          - Bian_xing        为源代码生成的变形红外图像

          - Fusion           为配准后的图像融合结果

          - Fusion_no_reg    为未配准的图像融合结果

          - Fusion_orignal   为源图像的融合结果

          - ir               为Roadscene源红外图像

          - ir_256           为进行Resize后红外图像

          - ir_256_map       为红外显著图

          - Peizhun          为配准后的红外图像

          - vi               为Roadscene源可见光图像

          - vi_256           为进行Resize后可见光图像

          - vi_256_map       为可见光显著图

          - Wei_hongwai      为源代码生成的伪红外图像
      
      Stage 3: 下载源作者给的测试权重 reg_0280.pth 和 fus_0280.pth

      Stage 4: 分别执行test_reg.py 和 test_fuse.py 生成配准图像和融合图像 （！！注意：本项目中给的配准结果和融合结果为自己训练后的权重文件，非原作者给的预训练权重）.
```
   2. 开源项目训练  
        （1）训练配准模型  
        ```python
           Stage 1: 在train_reg.py中更改参数配置，具体为
           parser.add_argument('--ir', default='dataset/Roadscene/ir_256', type=pathlib.Path) # 源红外图像
           parser.add_argument('--it', default='dataset/Roadscene/Wei_hongwai/rgb2ir_paired_Road_edge_pretrained/test_latest/images', type=pathlib.Path) # 伪红外图像

          Stage 2: 直接运行 python train_reg.py

          Stage 3: 训练结果在 cache\reg_0800.pth
       ```

       （2）训练融合模型  
       
         ```python
          Stage 1: train_fuse.py中更改参数配置，具体为

          parser.add_argument('--ir_reg', default='dataset/Roadscene/Peizhun/ir_reg', type=pathlib.Path) # 配准后的红外图像
          parser.add_argument('--vi',     default='dataset/Roadscene/vi_256', type=pathlib.Path) # 源红外图像
          parser.add_argument('--ir_map', default='dataset/Roadscene/ir_256_map', type=pathlib.Path)
          parser.add_argument('--vi_map', default='dataset/Roadscene/vi_256_map', type=pathlib.Path)

          Stage 2: 直接运行 python train_fuse.py

          Stage 3: 训练结果在 cache\fus_0200.pth

         ```

   3. 测试结果（均为自己训练的结果）  
       （1）配准测试结果（！！可以发现，训练后的模型配准还是有效果的）
       ![Fig\peizhun.png](https://github.com/HaodeShi/UMF-CMGR-retrain-and-test/blob/main/Fig/peizhun.png)  
       （2）配准定性测试结果
         ```python
            Stage 1: 运行 Evaluation\metrics.py，得到测试结果

            变形图像测试：Average MSE=0.010862, NCC=0.88515, LNCC=0.46209

            配准图像测试：Average MSE=0.0047011, NCC=0.94955, LNCC=0.58296
         ```

      （2）融合测试结果(可以观察到，配准与不配准融合差别还是挺大的)
       !![Fig\ronghe.png](https://github.com/HaodeShi/UMF-CMGR-retrain-and-test/blob/main/Fig/ronghe.png)  


## 3、主要参考资料  
### （1）参考论文
      ```
      @inproceedings{UMF,
         author    = {Di Wang and
         Jinyuan Liu and
         Xin Fan and
         Risheng Liu},
         title     = {Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration},
         booktitle = {IJCAI},
         pages     = {3508--3515},
         year      = {2022}
      }
      ```

### （2）https://github.com/wdhudiekou/UMF-CMGR

### （3）https://blog.csdn.net/qq_40280673/article/details/142585702
