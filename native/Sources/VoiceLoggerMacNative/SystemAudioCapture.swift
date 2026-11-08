import Foundation
import ScreenCaptureKit
import CoreMedia
import AVFoundation

/// In-process system-audio capture via ScreenCaptureKit (macOS 13+).
/// Port of `swift/SystemAudioCapture.swift`, which used to be a separate
/// helper process piping Float32 PCM to Python. Emits 16 kHz mono chunks.
@available(macOS 13.0, *)
public final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    public var onAudio: ([Float]) -> Void = { _ in }
    public var onError: (String) -> Void = { _ in }
    public var onStatus: (String) -> Void = { _ in }

    private var stream: SCStream?
    private let audioQueue = DispatchQueue(label: "voicelogger.audio", qos: .userInteractive)
    private let sampleRate = 16000

    public func start() async throws {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            onError("Sin permiso de grabación de pantalla/audio: \(error.localizedDescription)")
            throw error
        }
        guard let display = content.displays.first else {
            let err = NSError(domain: "VoiceLogger", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "No se encontró ninguna pantalla para capturar"])
            onError(err.localizedDescription)
            throw err
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = sampleRate
        config.channelCount = 1
        config.excludesCurrentProcessAudio = true
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false
        config.queueDepth = 8

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        try await stream.startCapture()
        self.stream = stream
        onStatus("Escuchando audio del sistema...")
    }

    public func stop() async {
        if let stream {
            try? await stream.stopCapture()
            self.stream = nil
        }
    }

    public func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        var blockBuffer: CMBlockBuffer?
        var audioBufferList = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )
        guard status == noErr else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(&audioBufferList)
        for buffer in buffers {
            guard let data = buffer.mData, buffer.mDataByteSize > 0 else { continue }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            let floats = UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: count)
            onAudio(Array(floats))
        }
    }

    public func stream(_ stream: SCStream, didStopWithError error: Error) {
        onError("macOS detuvo la captura: \(error.localizedDescription)")
    }
}
