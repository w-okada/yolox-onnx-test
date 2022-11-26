const ort = require("onnxruntime-web");
const modelFile = require("../models/yolox_nano_416x416.onnx");
async function main() {
    try {
        const session = await ort.InferenceSession.create(modelFile);

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
    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

main();
