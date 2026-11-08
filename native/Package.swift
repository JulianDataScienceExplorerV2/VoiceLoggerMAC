// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VoiceLoggerMacNative",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "VoiceLoggerMacNative", targets: ["VoiceLoggerMacNative"])
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", from: "1.1.0")
    ],
    targets: [
        .executableTarget(
            name: "VoiceLoggerMacNative",
            dependencies: [.product(name: "WhisperKit", package: "argmax-oss-swift")],
            path: "Sources/VoiceLoggerMacNative"
        ),
        .testTarget(
            name: "VoiceLoggerMacNativeTests",
            dependencies: ["VoiceLoggerMacNative"],
            path: "Tests/VoiceLoggerMacNativeTests"
        ),
    ]
)
