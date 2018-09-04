# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)
---------------------------------------------------------------------------------------------------------
## MOBILENET-SSD的使用  
通过使用mobilenet-ssd为例子，学会使用google的API  
--------------------------------------------------
（1）转换训练样本成tfrecord格式，需要提前将数据按照VOC数据集格式整理，制作label_map。
将需要转化的图片放在models/research/object_detection/test_changshidata  
转换训练数据和测试数据：  
python create_pascal_tf_record.py   
--data_dir=/home/liuyp/liu/models/VOCdevkit/   --label_map_path=/home/liuyp/liu/models/research/object_detection/data/skateboard.pbtxt --set=val   
--output_path=/home/liuyp/liu/models/tfrecoard_data/val_skateboard.record   
--year=VOC2012    
（2）进行训练  
预训练模型放在地址：/models/research/object_detection/ssd_mobilenet_v1_coco_2018_01_28    
运行代码 ：  
python train.py   
--logtostderr   
--train_dir=/home/liuyp/liu/models/research/object_detection/model/ssd_mobilenet_v1_coco_2018_01_28/save_model   
--pipeline_config_path=/home/liuyp/liu/models/research/object_detection/model/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config  
（3）模型转换  
python export_inference_graph.py     
--input_type image_tensor    
--pipeline_config_path=/models/research/object_detection/model/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config   --trained_checkpoint_prefix=/models/research/object_detection/model/ssd_mobilenet_v1_coco_2018_01_28/save_model/model.ckpt-12923 
--output_directory=/models/research/object_detection/model/ssd_mobilenet_v1_coco_2018_01_28/saved_model  
最后输出.pb模型存放在：  
/models/research/object_detection/ssd_mobilenet_v1_coco_2018_01_28/saved_model  
