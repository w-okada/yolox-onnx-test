YOLOX-ONNX-TEST
----
This repository is intended only to evaluate the performance of the yolox onnx model. 

We evaluated from two perspectives.

1. Python vs Wasm
2. Wasm vs TFJS on Browser 

# Prerequisite
This repository use docker.

# Experiment 1
In this experiment, we used dummy data. Below are the condition and the results of the experiment.

## python onnx (cpu)
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
for i in range(10):
    results = onnx_session.run(None, {"images": dummy},)
elapsed_time = time.time() - start_time
print(size, 'fp32 fin. avr time:', (elapsed_time / 10) * 1000, "msec")
```
Output is "`fin. avr time: 8.36 msec`" on "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"

## onnx runtime web sample (wasm)
This code is used.
```js
const start = performance.now();
let results = "";
const im = Float32Array.from(Array(1 * 3 * 416 * 416).fill(0));
const tensorIm = new ort.Tensor("float32", im, [1, 3, 416, 416]);
const feeds = { images: tensorIm };
for (let i = 0; i < 10; i++) {
    results = await session.run(feeds);
}
const end = performance.now();

console.log(results);
document.write(`fin. Avr. ${((end - start) / 10).toFixed(2)}msec`);
```
Output is "`fin. Avr. 67.92msec`" on "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"


## Result

Python onnx is faster than onnx runtime web 8 times.

| #   | onnx runtime web sample (wasm) | python onnx (cpu) |
| --- | ------------------------------ | ----------------- |
| Avr | 67.92msec                      | 8.36msec          |

# Experiment 2
ONNX runtime web with WEBGL does not seem to support float16 and int64.
[issue](https://github.com/microsoft/onnxruntime/issues/9724)

So, I convert onnx to tfjs(Tensorflowjs) and test it.

Below are the condition and the results of the experiment.

## ONNX runtime web (wasm)

![image](https://user-images.githubusercontent.com/48346627/204077770-4bf0f56e-6d2e-491c-85fa-a1ec0e1b1240.png)

All(include pre-process and post-process): about 75ms

Inference only: about 65ms

on "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"

## Tensorflowjs

![image](https://user-images.githubusercontent.com/48346627/204077788-db62abeb-2877-4351-8d89-5ea5e1755b8a.png)

All(include pre-process and post-process): about 17ms

Inference only: about 11ms

on "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz" and "NVIDIA GeForce RTX 2080 Ti"

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

## Web.
Access [here](https://w-okada.github.io/yolox-onnx-test/)

# Reference
1. https://github.com/Megvii-BaseDetection/YOLOX
1. https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample

