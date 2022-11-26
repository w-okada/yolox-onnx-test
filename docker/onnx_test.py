import script.demo as demo
import numpy as np
import onnxruntime
import time

img_sizes = ["416x416", "320x320", "640x640", "1280x1280", "1920x1920", "256x320",
             "256x480", "256x640", "384x640", "480x640", "736x1280", "1088x1920"]
#img_sizes = ["1088x1920"]
# img_sizes = ["416x416"]

for size in img_sizes:
    h, w = size.split("x")
    print(f"processing... {size}")
    demo.process(f"/models/yolox_nano_{size}.onnx", "/data/D0002011239_00000.jpg",
                 f"/data/output/yolox-onnx-{size}.jpg", input_shape=(int(h), int(w)))

for size in img_sizes:
    h, w = size.split("x")
    input_shape = (int(h), int(w))
    providers = ['CPUExecutionProvider']

    model_path = f"/models/yolox_nano_{size}.onnx"
    dummy = np.zeros((1, 3, int(h), int(w))).astype(np.float32)
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
    )
    start_time = time.time()
    for i in range(10):
        results = onnx_session.run(None, {"images": dummy},)
    elapsed_time = time.time() - start_time
    print(size, 'fp32 fin. avr time:', (elapsed_time / 10) * 1000, "msec")
