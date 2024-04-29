# 练习3

#### 下采样

```python
from PIL import Image
import os

path='set5'
save_path='down_bicubic'
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    
    # 打开原始图像
    image = Image.open(file_path)
    
    # 指定目标尺寸
    target_width = int(image.width * 0.5)  # 50% 的宽度
    target_height = int(image.height * 0.5)  # 50% 的高度

    # 进行 Bicubic 下采样
    resized_image = image.resize((target_width, target_height), Image.BICUBIC)

    # 保存处理后的图像
    resized_image.save(os.path.join(save_path,file))
```

#### 超分

1、下载预训练模型

下载 stable-diffusion-2-base模型  [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).

下载 SeeSR and DAPE 模型  [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link) or [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/22042244r_connect_polyu_hk/EiUmSfWRmQFNiTGJWs7rOx0BpZn2xhoKN6tXFmTSGJ4Jfw?e=RdLbvg).

2、准备测试数据

将测试数据放入`preset/datasets/test_datasets`

3、运行测试命令

```
python test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path preset/models/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 
```



#### 对比

```python
import cv2
from skimage.metrics import structural_similarity as ssim

origin_path ='set5'
sr_path='sp_result'
for file in os.listdir(origin_path):
    # 加载图像
    origin_filepath = os.path.join(origin_path, file)
    sr_filepath=os.path.join(sr_path,file)
    origin_image=cv2.imread(origin_filepath)
    sr_image=cv2.imread(sr_filepath)
    # 统一图像尺寸
    origin_image=cv2.resize(origin_image, (sr_image.shape[1], sr_image.shape[0]))
    print(file)
    # PSNR
    psnr = cv2.PSNR(origin_image, sr_image)
    print("PSNR:", psnr)
    # SSIM
    ssim_value, _ = ssim(origin_image, sr_image, full=True, multichannel=True,channel_axis=2)
    print("SSIM:", ssim_value)
```



#### 结果

<img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/set5/baby.png" alt="baby" width="160px"/><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/down_bicubic/baby.png" alt="baby" width="160px"/><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/sp_result/baby.png" alt="baby" width="160px"/>

<img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/set5/bird.png" alt="bird" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/down_bicubic/bird.png" alt="bird" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/sp_result/bird.png" alt="bird" width="160px" />

<img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/set5/butterfly.png" alt="butterfly" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/down_bicubic/butterfly.png" alt="butterfly" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/sp_result/butterfly.png" alt="butterfly" width="160px" />


<img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/set5/head.png" alt="head" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/down_bicubic/head.png" alt="head"  width="160px"/><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/sp_result/head.png" alt="head" width="160px" />



<img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/set5/woman.png" alt="woman" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/down_bicubic/woman.png" alt="woman" width="160px" /><img src="https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/sp_result/woman.png" alt="woman" width="160px" />





|           | PSNR               | SSIM               |
| --------- | ------------------ | ------------------ |
| baby      | 25.406378545999754 | 0.7545208215131695 |
| bird      | 20.797982562277607 | 0.6533684526798802 |
| butterfly | 16.625738844683553 | 0.617332435000412  |
| head      | 26.62857969703666  | 0.6998981997423286 |
| woman     | 20.447409041764825 | 0.7198145143506719 |





## SeeSR

![framework](https://github.com/wtgwuhoo/Computer-Vision-Practice-Exercise-1/tree/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A03/SeeSR/figs/framework.png)

这个系统分为两个阶段：

(a) **降质感知提示提取器（DAPE）**：这是训练过程的第一阶段。在这里，一个高分辨率（HR）图像被随机降质成低分辨率（LR）图像。这个过程涉及两个图像编码器——一个用于HR图像，另一个用于LR图像。两者通过相应的标签头输出标签嵌入，这些标签嵌入用于初始化并通过代表性损失和逻辑损失进行训练。目标是让DAPE能够识别出降质图像与原始HR图像之间的关系，并能够提取有助于后续恢复过程的提示。

(b) **带DAPE的Real-ISR**：训练好的DAPE在第二阶段用于实际图像超分辨率（Real-ISR）。DAPE提供软提示（代表性嵌入）和硬提示（标签文本），结合LR图像，这些提示用来控制预先训练的T2I（Text-to-Image）扩散模型。这个模型的目的是恢复LR图像，使其尽可能接近原始的HR图像。

(c) **受控T2I扩散模型**：这是系统的核心，一个详细的结构用于处理图像和文本提示。这个模型有三个分支：图像分支、文本分支和代表性分支。模型利用文本编码器和Unet编码器，通过文本交叉注意（TCA）和代表性交叉注意（RCA）模块，将文本和图像特征融合在一起，最后通过Unet解码器和控制网络生成恢复的图像。

整个系统的目标是提高图像超分辨率的性能，特别是在处理降质图像时，通过结合文本和图像的提示来控制生成过程，以此提供更精确和细致的结果。通过这种方式，SeeSR能够生成视觉上令人满意且细节丰富的高分辨率图像。
