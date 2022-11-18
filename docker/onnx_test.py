import onnxruntime
import numpy as np
import time
model_path = "/work/yolox_nano.onnx"
providers=['CPUExecutionProvider']
onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
#print(onnx_session.get_inputs()[0].name)
dummy = np.zeros((1,3,416,416)).astype(np.float32)
start_time = time.time()
for i in range(100):
    results = onnx_session.run(None,{"images": dummy},)
elapsed_time = time.time() - start_time
print('fin. avr time:', (elapsed_time / 100) * 1000, "msec")
#print(results[0].shape)


