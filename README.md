YOLOX-ONNX-TEST
----
This repository is intended only to evaluate the performance of the yolox onnx model. Only dummy data is used for inference.


1. onnx runtime web
2. python onnx (TBD)

# Prerequisite
This repository use docker.

# onnx runtime sample
This code is used.
```js
const start = performance.now();
let results = "";
const im = Float32Array.from(Array(1 * 3 * 416 * 416).fill(0));
const tensorIm = new ort.Tensor("float32", im, [1, 3, 416, 416]);
const feeds = { images: tensorIm };
for (let i = 0; i < 100; i++) {
    results = await session.run(feeds);
}
const end = performance.now();

console.log(results);
document.write(`fin. Avr. ${((end - start) / 100).toFixed(2)}msec`);
```
Output is "`fin. Avr. 67.92msec`" on "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"

# python onnx
This code is used.
```py
import onnxruntime
import numpy as np
import time
model_path = "/work/yolox_nano.onnx"
providers=['CPUExecutionProvider']
onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
dummy = np.zeros((1,3,416,416)).astype(np.float32)
start_time = time.time()
for i in range(100):
    results = onnx_session.run(None,{"images": dummy},)
elapsed_time = time.time() - start_time
print('fin. avr time:', (elapsed_time / 100) * 1000, "msec")
```
Output is "`fin. avr time: 8.36 msec`" on "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"


# Operations
You can reproduce the experiment according to the following way.

## build docker
```
$ npm run build:docker
```

## export onnx model
Only `yolox_nano.pth` is supported.
```
$ npm run export:model
```

## build onnx runtime sample and run.
```
$ npm install 
$ npm run build:web
$ npm run watch
```
Access to the url shown in terminal and wait for a while. Then you can see the output on the browser.

## run python onnx
```
$ npm run start:python
```
Wait for a while. Then you can see the output on the termianl.

# Reference
1. https://github.com/Megvii-BaseDetection/YOLOX
1. https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample

