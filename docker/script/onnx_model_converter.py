
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import argparse

# (1) FP16 make no improve with CPU.(performance. but file size is down)


def _convert_float_to_float16_model_path(input_file_path, output_file_path):
    new_onnx_model = convert_float_to_float16_model_path(input_file_path)
    onnx.save(new_onnx_model, output_file_path)


# (2) No improve with Convolution
# (2-1) https://github.com/microsoft/onnxruntime/issues/3130
def _quantize_dynamic_ui8(input_file_path, output_file_path):
    quantize_dynamic(input_file_path, output_file_path,
                     weight_type=QuantType.QUInt8)

# (2-2) only linear function yolox have too few linear function to improve??


def _quantize_dynamic_i8op(input_file_path, output_file_path):
    model = onnx.load(input_file_path)
    nodes = model.graph.node
    names = [x.name for x in nodes]

    prefix = ["MatMul", "Add", "Relu"]
    linear_names = [v for v in names if v.split("_")[0] in prefix]
    quantize_dynamic(input_file_path, output_file_path, weight_type=QuantType.QInt8,
                     nodes_to_quantize=linear_names, extra_options={"MatMulConstBOnly": True})

# (2-3) only linear function yolox have too few linear function to improve??


def _quantize_dynamic_ui8op(input_file_path, output_file_path):
    model = onnx.load(input_file_path)
    nodes = model.graph.node
    names = [x.name for x in nodes]

    # prefix = ["MatMul", "Add", "Relu"]
    prefix = ["Conv"]
    linear_names = [v for v in names if v.split("_")[0] not in prefix]

    quantize_dynamic(input_file_path, output_file_path, weight_type=QuantType.QUInt8,
                     nodes_to_quantize=linear_names, extra_options={"MatMulConstBOnly": True})


# import onnx

# model = onnx.load("/models/yolox_nano_1088x1920_i8.onnx")

# nodes = model.graph.node
# names = [x.name for x in nodes]
# print(nodes)
# prefix = ["Conv"]
# linear_names = [v for v in names if v.split("_")[0] not in prefix]
# print(linear_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str,
                        help="input model file")
    parser.add_argument("-o", "--output_file", type=str,
                        help="output model file")
    parser.add_argument("-t", "--type", type=str,
                        help="covert type <fp16|quantize>")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    if args.type == "fp16":
        print("converting model to fp16...")
        _convert_float_to_float16_model_path(input_file, output_file)
    else:
        print("quantizing model...")
        # _quantize_dynamic_ui8(input_file, output_file)
        #_quantize_dynamic_i8op(input_file, output_file)
        _quantize_dynamic_ui8op(input_file, output_file)
