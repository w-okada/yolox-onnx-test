YOLOX-ONNX-TEST
----
This repository is intended only to evaluate the performance of the yolox onnx model. Only dummy data is used for inference.


1. onnx runtime web
2. python onnx (TBD)

# Prerequisite
This repository use docker.

# Operations
## export onnx model
Only `yolox_nano.pth` is supported.
```
$ npm run build:docker
$ npm run export:model
```

## build onnx runtime sample
```
$ npm install 
$ npm run build:web
$ npm run watch
```

# Reference
1. https://github.com/Megvii-BaseDetection/YOLOX
1. https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample

