import Foundation
import ScreenCaptureKit
import CoreMedia
import AVFoundation

let stdoutHandle = FileHandle.standardOutput
let stderrHandle = FileHandle.standardError

func logError(_ message: String) {
    if let data = (message + "\n").data(using: .utf8) {
        stderrHandle.write(data)
    }
}

func installSignalHandlers() {
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        signal(sig, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        source.setEventHandler {
            stdoutHandle.synchronizeFile()
            exit(0)
        }
        source.resume()
    }
}

@available(macOS 13.0, *)
final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let audioQueue = DispatchQueue(label: "voicelogger.audio", qos: .userInteractive)
    private var formatLogged = false
    private let sampleRate = 16000

    func start() async throws {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            logError("SCREEN_RECORDING_PERMISSION_DENIED: \(error.localizedDescription)")
            throw error
        }

        guard let display = content.displays.first else {
            // Antes hacía `return` y el proceso se quedaba vivo sin capturar nada.
            logError("NO_DISPLAY_FOUND")
            throw NSError(
                domain: "VoiceLogger", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No se encontró ninguna pantalla para capturar"]
            )
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
        logError("CAPTURE_STARTED rate=\(sampleRate) channels=1")
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }

        if !formatLogged, let format = CMSampleBufferGetFormatDescription(sampleBuffer),
           let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee {
            logError("AUDIO_FORMAT rate=\(asbd.mSampleRate) channels=\(asbd.mChannelsPerFrame) bits=\(asbd.mBitsPerChannel) flags=\(asbd.mFormatFlags) bytesPerFrame=\(asbd.mBytesPerFrame)")
            formatLogged = true
        }

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
            guard let data = buffer.mData else { continue }
            let byteCount = Int(buffer.mDataByteSize)
            if byteCount > 0 {
                // `write(contentsOf:)` lanza un error Swift si el lector se cierra
                // (EPIPE); `write(_:)` lanzaba una excepción ObjC que mataba el proceso.
                do {
                    try stdoutHandle.write(contentsOf: Data(bytes: data, count: byteCount))
                } catch {
                    exit(0)
                }
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        logError("STREAM_STOPPED_WITH_ERROR: \(error.localizedDescription)")
        exit(1)
    }
}

@main
struct Main {
    static func main() async {
        installSignalHandlers()

        if #available(macOS 13.0, *) {
            let capture = SystemAudioCapture()
            do {
                try await capture.start()
            } catch {
                logError("CAPTURE_FAILED: \(error.localizedDescription)")
                exit(1)
            }
            while true {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        } else {
            logError("UNSUPPORTED_MACOS: se requiere macOS 13 o superior")
            exit(1)
        }
    }
}
