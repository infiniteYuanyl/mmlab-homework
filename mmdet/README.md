# mmdet作业说明

### 1.基础作业

自定义数据集类型BalloonDataset，注册后进行训练，使用预训练模型`mask_rcnn_r50_fpn_mstrain-poly_3x_coco.pth`,训练了48轮后停止。测试mAP值良好，基本在50以上。

![image-20230208140816232](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230208140816232.png)

splash脚本编写如下，放在tools文件夹下，主要使用opencv和skimage库。

`color_splash.py`

```python
import skimage.draw

import  argparse
import datetime
import  numpy as np

# Root directory of the project
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
# register all modules in mmdet into the registries
register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='splash color from vidio')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('video_path', help='video path')
    args = parser.parse_args()

    return args
# Import Mask RCNN
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [instance count,height,width]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # put mask on cpu and change its dims
    # from [instance count,height,width] to [height,width,instance count]
    mask = mask.cpu().numpy()
    mask = np.transpose(mask, [1, 2, 0])
    # We're treating all instances as one, so collapse the mask into one layer
    
    mask = (np.sum(mask, axis=-1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model,  video_path=None):
    assert video_path
    # Image or video?
    import cv2
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))

    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects

            r = inference_detector(model, image)
            # r = model.detect([image], verbose=0)[0]
            # Color splash
            splash = color_splash(image, r.pred_instances['masks'])
            # RGB -> BGR to save image to video
            splash = splash[..., ::-1]
            # Add image to video writer
            vwriter.write(splash)
            count += 1
    vwriter.release()
    print("Saved to ", file_name)
if __name__ == '__main__':

     # usage: python tools/color_splash.py config checkpoint video_path
     # eg: python tools/color_splash.py work_dirs/mask-rcnn_r50_fpn_ms-poly-3x_balloon/mask-rcnn_r50_fpn_ms-poly-3x_balloon.py work_dirs/mask-rcnn_r50_fpn_ms-poly-3x_balloon/epoch_48.pth test_video.mp4


    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    detect_and_color_splash(model,video_path=args.video_path)

```



生成视频效果截屏如下：

![image-20230208141316575](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230208141316575.png)

训练模型文件链接：

链接：https://pan.baidu.com/s/1DM6burkdPOcWH33Ta5ItNA 
提取码：tw41 

视频效果文件链接：

链接：https://pan.baidu.com/s/18f0kbxV9JxTgIvpaCczuyg 
提取码：qqjs 



### 2.进阶作业

使用`fast_rcnn`模型完成VOC2012的任务。模型配置文件放在advance目录下，日志文件在日期目录下。

验证指标如下，效果比较不错

![image-20230210020000711](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgimage-20230210020000711.png)

![image-20230210020016241](C:/Users/yuan/AppData/Roaming/Typora/typora-user-images/image-20230210020016241.png)



将demo视频转换后，制作gif效果如下

![pascal_voc2012](https://yuan-1314071695.cos.ap-nanjing.myqcloud.com/imgpascal_voc2012.gif)

训练模型文件链接：

链接：https://pan.baidu.com/s/1T3D1eoknP7D_nXA9A1LMKQ 
提取码：x3c8 




