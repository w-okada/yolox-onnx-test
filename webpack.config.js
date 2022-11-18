// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
    mode: "development",
    // mode: "production",
    target: ["web"],
    entry: path.resolve(__dirname, "web", "main.js"),
    output: {
        path: path.resolve(__dirname, "web", "dist"),
        filename: "bundle.min.js",
        library: {
            type: "umd",
        },
    },
    module: {
        rules: [{ test: /\.onnx$/, type: "asset/resource" }],
    },
    plugins: [
        new CopyPlugin({
            patterns: [
                {
                    from: "node_modules/onnxruntime-web/dist/*.wasm",
                    to: "[name][ext]",
                },
            ],
        }),
    ],
    devServer: {
        static: {
            directory: path.join(__dirname, "web"),
        },
        client: {
            overlay: {
                errors: false,
                warnings: false,
            },
        },
    },
};
