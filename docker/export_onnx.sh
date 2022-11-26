
img_sizes=(416x416 320x320 640x640 1280x1280 1920x1920 256x320 256x480 256x640 384x640 480x640 736x1280 1088x1920)
#img_sizes=(1088x1920)
#img_sizes=(416x416)

for size in ${img_sizes[@]}; do
    echo $size > resolution
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. python3 tools/export_onnx.py -n yolox-nano -c yolox_nano.pth -f script/export_onnx_nano_conf.py --output-name /models/yolox_nano_${size}.onnx
    
    onnx2tf -i /models/yolox_nano_${size}.onnx -o /models/tflite_yolox_nano_${size} --output_signaturedefs
    
    tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default  --saved_model_tags=serve /models/tflite_yolox_nano_${size} /models/tfjs/tfjs_yolox_nano_${size}

    # FP16 make bad effect for CPU??(only file size id down,  performance is worse)
    # python3 script/onnx_model_converter.py -i /models/yolox_nano_${size}.onnx -o /models/yolox_nano_${size}_fp16.onnx -t fp16
    # I64 to I32 not work in ONNX Runtime Web..
    # python3 script/convert_i64_to_i32.py -i /models/yolox_nano_${size}.onnx -o /models/yolox_nano_${size}_i32.onnx 

    # Quantization make no improve for CPU??
    # python3 script/onnx_model_converter.py -i /models/yolox_nano_${size}.onnx -o /models/yolox_nano_${size}_i8.onnx -t quantize
done
