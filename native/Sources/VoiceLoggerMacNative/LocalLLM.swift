import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Runs the language model inside the app with MLX Swift (no Python).
///
/// Same MLX 4-bit weights and the same Hugging Face cache as the Python
/// engine (`~/.cache/huggingface/hub`), so models already downloaded are reused.
/// Memory: prompt prefill in 512-token steps, MLX buffer cache capped at
/// 512 MB and emptied after each answer (as in `summarizer.py`).
actor LocalLLM {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static let prefillStep = 512
    static let cacheLimitBytes = 512 * 1024 * 1024
    static let repetitionPenalty: Float = 1.1
    static let repetitionContext = 64
    /// Estimate when the tokenizer is not loaded yet.
    static let charsPerToken = 3.5

    private(set) var modelID: String
    private var container: ModelContainer?
    private var loading: Task<ModelContainer, Error>?
    /// One generation at a time, in arrival order (like the single model
    /// thread of summarizer.py): the GPU is never shared by two answers.
    private var isGenerating = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    init(modelID: String) {
        self.modelID = modelID
    }

    var isLoaded: Bool { container != nil }

    /// The next request loads `id` (the current model is freed).
    func setModel(_ id: String) {
        guard id != modelID else { return }
        modelID = id
        unload()
    }

    /// Frees the model from memory; it reloads on the next request.
    @discardableResult
    func unload() -> Bool {
        let wasLoaded = container != nil || loading != nil
        loading?.cancel()
        loading = nil
        container = nil
        Memory.clearCache()
        return wasLoaded
    }

    func load() async throws -> ModelContainer {
        if let container { return container }
        if let loading { return try await loading.value }
        guard Self.metalLibraryAvailable() else {
            throw Failure(message: "Esta compilación no trae los shaders de MLX (default.metallib). "
                + "Empaqueta la app con native/package.sh (usa xcodebuild; «swift build» no los compila) "
                + "o vuelve al motor Python en ⚙.")
        }
        Memory.cacheLimit = Self.cacheLimitBytes
        let id = modelID
        let configuration = ModelConfiguration(id: id)
        let task = Task { try await #huggingFaceLoadModelContainer(configuration: configuration) }
        loading = task
        do {
            let loaded = try await task.value
            guard id == modelID else { throw CancellationError() }  // the model changed while loading
            container = loaded
            loading = nil
            return loaded
        } catch {
            if id == modelID { loading = nil }
            throw error
        }
    }

    private func acquire() async {
        if isGenerating {
            await withCheckedContinuation { waiting.append($0) }  // the finishing one hands over its turn
        } else {
            isGenerating = true
        }
    }

    private func release() {
        if waiting.isEmpty {
            isGenerating = false
        } else {
            waiting.removeFirst().resume()
        }
    }

    /// One answer: `system` + `user` → text without reasoning or code fences.
    func generate(
        system: String, user: String, maxTokens: Int, temperature: Float = 0.3, penalizeRepeats: Bool = true
    ) async throws -> String {
        await acquire()
        defer { release() }
        let container = try await load()
        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            repetitionPenalty: penalizeRepeats ? Self.repetitionPenalty : nil,
            repetitionContextSize: Self.repetitionContext,
            prefill: .init(stepSize: Self.prefillStep)
        )
        // One fresh session per request: system + one user message, as in Python.
        // Qwen3 "thinks" by default; live updates only need the answer.
        let session = ChatSession(
            container, instructions: system, generateParameters: parameters,
            additionalContext: ["enable_thinking": false]
        )
        defer { Memory.clearCache() }
        let raw = try await session.respond(to: user)
        return AIText.stripCodeFence(AIText.stripThinking(raw))
    }

    /// Token count with the model's tokenizer (an estimate if it is not loaded).
    func countTokens(_ text: String) async -> Int {
        guard let container else { return Int(Double(text.count) / Self.charsPerToken) + 1 }
        return await container.encode(text).count
    }

    /// Cuts `text` to `maxTokens` without breaking lines: `keepTail` keeps the
    /// end (the most recent part), otherwise the beginning (notes start with
    /// the summary). Port of `fit_tokens`.
    func fitTokens(_ text: String, _ maxTokens: Int, keepTail: Bool) async -> String {
        var text = AIText.trimmed(text)
        for _ in 0..<4 {
            let count = await countTokens(text)
            if count <= maxTokens { return text }
            let cut = max(1, Int(Double(text.count) * Double(maxTokens) / Double(count) * 0.95))
            if keepTail {
                text = String(text.suffix(cut))
                if let newline = text.firstIndex(of: "\n"), text.index(after: newline) < text.endIndex {
                    text = String(text[text.index(after: newline)...])
                }
            } else {
                text = String(text.prefix(cut))
                if let newline = text.lastIndex(of: "\n"), newline > text.startIndex {
                    text = String(text[..<newline])
                }
            }
        }
        return text
    }

    /// MLX needs its Metal shaders (default.metallib, inside the
    /// mlx-swift_Cmlx bundle). `swift build` does not compile them; xcodebuild
    /// (native/package.sh) does. Without them MLX would stop the app.
    static func metalLibraryAvailable() -> Bool {
        let fm = FileManager.default
        var roots: [URL] = []
        if let resources = Bundle.main.resourceURL { roots.append(resources) }
        if let executable = Bundle.main.executableURL?.deletingLastPathComponent() { roots.append(executable) }
        for root in roots {
            guard let items = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { continue }
            for item in items {
                if item.pathExtension == "metallib" { return true }
                guard item.pathExtension == "bundle" || item.pathExtension == "framework" else { continue }
                let candidates = [
                    item.appendingPathComponent("default.metallib"),
                    item.appendingPathComponent("Contents/Resources/default.metallib"),
                    item.appendingPathComponent("Resources/default.metallib"),
                    item.appendingPathComponent("mlx.metallib"),
                ]
                if candidates.contains(where: { fm.fileExists(atPath: $0.path) }) { return true }
            }
        }
        return false
    }
}
