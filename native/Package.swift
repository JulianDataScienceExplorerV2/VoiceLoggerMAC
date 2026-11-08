// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "VoiceLoggerMacNative",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "VoiceLoggerMacNative", targets: ["VoiceLoggerMacNative"])
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", from: "1.1.0"),
        // Native AI engine ("Swift nativo" in ⚙): MLX Swift runs the model inside
        // the app; swift-huggingface downloads it (same cache as Python) and
        // swift-transformers provides the tokenizer. MLX's Metal shaders are only
        // built by xcodebuild (native/package.sh), not by `swift build`.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        // The FoundationModels bridge (default trait) needs the macOS 27 SDK and is not used.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3"), traits: []),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "VoiceLoggerMacNative",
            dependencies: [
                .product(name: "WhisperKit", package: "argmax-oss-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/VoiceLoggerMacNative"
        ),
        .testTarget(
            name: "VoiceLoggerMacNativeTests",
            dependencies: ["VoiceLoggerMacNative"],
            path: "Tests/VoiceLoggerMacNativeTests"
        ),
    ]
)
