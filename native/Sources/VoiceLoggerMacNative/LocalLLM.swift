import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Text so far of an answer being written (to show it while it is generated).
typealias PartialText = @Sendable (String) -> Void

/// Runs the language model inside the app with MLX Swift (no Python).
///
/// Same MLX 4-bit weights and the same Hugging Face cache as the Python
/// engine (`~/.cache/huggingface/hub`), so models already downloaded are reused.
///
/// Optimizations (things the Python engine did not do):
/// 1. **Streaming**: the answer is passed on while it is written (`partial`),
///    so the first words show up in about a second.
/// 2. **Speculative decoding**: a small model of the same family proposes a few
///    tokens and the main model checks them in one pass. Same output quality,
///    faster writing; ~0.4 GB more RAM. Off if the helper cannot be loaded.
/// 3. **Prompt prefix cache**: the system prompt (instructions + context file)
///    is prefilled once per variant; later requests start from a copy of that
///    cache (`ChatSession.fork`) and only prefill their own text.
/// 4. **8-bit KV cache** past 1024 tokens: about half the working memory of long
///    prompts (live summary, answers with web pages), nearly lossless.
///
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
    /// Qwen3 "thinks" by default; live updates only need the answer.
    static let chatContext: [String: any Sendable] = ["enable_thinking": false]

    /// Helper ("draft") model per main model: same tokenizer, much smaller.
    static let draftModels: [String: String] = [
        "mlx-community/Qwen3-4B-Instruct-2507-4bit": "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    ]
    /// Tokens the helper proposes per check (3 works best for small 4-bit models).
    static let numDraftTokens = 3
    static let kvBits = 8
    /// Short prompts (live questions) keep a full-precision cache.
    static let quantizedKVStart = 1024
    /// System prompt variants kept prefilled (meeting / ideas, per language).
    static let maxCachedPrefixes = 3
    /// Seconds between partial-text updates (the UI does not need every token).
    static let partialInterval: Duration = .milliseconds(100)

    private(set) var modelID: String
    private var useDraft: Bool
    private var container: ModelContainer?
    private var loading: Task<ModelContainer, Error>?
    private var draft: ModelContainer?
    /// The helper could not be loaded (e.g. offline the first time): not retried
    /// on every request, only after a model change or unload.
    private var draftUnavailable = false
    /// Sessions whose cache already holds a system prompt, most recent last.
    private var prefixes: [(key: String, session: ChatSession)] = []
    /// An optimized generation failed: plain generation (no helper, prefix
    /// cache or 8-bit cache) until the model changes or is unloaded.
    private var safeMode = false
    /// One generation at a time, in arrival order (like the single model
    /// thread of summarizer.py): the GPU is never shared by two answers.
    private var isGenerating = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    init(modelID: String, useDraft: Bool = true) {
        self.modelID = modelID
        self.useDraft = useDraft
    }

    var isLoaded: Bool { container != nil }

    /// The next request loads `id` (the current model is freed).
    func setModel(_ id: String) {
        guard id != modelID else { return }
        modelID = id
        unload()
    }

    /// Turns speculative decoding on or off (frees the helper when off).
    func setDraftEnabled(_ enabled: Bool) {
        guard enabled != useDraft else { return }
        useDraft = enabled
        draft = nil
        draftUnavailable = false
        prefixes = []
        safeMode = false
        Memory.clearCache()
    }

    /// Frees the models from memory; they reload on the next request.
    @discardableResult
    func unload() -> Bool {
        let wasLoaded = container != nil || loading != nil
        if wasLoaded { SessionLog.shared.log("IA", "modelo liberado de la memoria") }
        loading?.cancel()
        loading = nil
        container = nil
        draft = nil
        draftUnavailable = false
        prefixes = []
        safeMode = false
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
        let started = Date()
        SessionLog.shared.log("IA", "Swift: cargando \(id)…")
        let task = Task { try await #huggingFaceLoadModelContainer(configuration: configuration) }
        loading = task
        do {
            let loaded = try await task.value
            guard id == modelID else { throw CancellationError() }  // the model changed while loading
            container = loaded
            loading = nil
            SessionLog.shared.log(
                "IA", "Swift: \(id) listo en \(SessionLog.seconds(Date().timeIntervalSince(started))) · "
                    + "MLX \(Self.megabytes(Memory.activeMemory)) MB · app \(SessionLog.memoryMB()) MB"
            )
            return loaded
        } catch {
            if id == modelID { loading = nil }
            SessionLog.shared.log("ERROR", "Swift: no se pudo cargar \(id): \(error.localizedDescription)")
            throw error
        }
    }

    /// The helper model for speculative decoding, or nil (off, no helper for
    /// this model, or it could not be loaded: then generation is just normal).
    private func draftModel() async -> ModelContainer? {
        guard useDraft, !draftUnavailable, let id = Self.draftModels[modelID] else { return nil }
        if let draft { return draft }
        let main = modelID
        let started = Date()
        do {
            let loaded = try await #huggingFaceLoadModelContainer(configuration: ModelConfiguration(id: id))
            guard main == modelID, useDraft else { return nil }
            draft = loaded
            SessionLog.shared.log(
                "IA", "ayudante \(id) listo en \(SessionLog.seconds(Date().timeIntervalSince(started))) "
                    + "(decodificación especulativa, \(Self.numDraftTokens) tokens por paso)"
            )
            return loaded
        } catch {
            if main == modelID { draftUnavailable = true }
            SessionLog.shared.log("IA", "ayudante \(id) no disponible, sigue sin él: \(error.localizedDescription)")
            return nil
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

    private func makeParameters(
        maxTokens: Int, temperature: Float, penalizeRepeats: Bool, quantizeKV: Bool = true
    ) -> GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            kvBits: quantizeKV ? Self.kvBits : nil,
            quantizedKVStart: quantizeKV ? Self.quantizedKVStart : 0,
            temperature: temperature,
            repetitionPenalty: penalizeRepeats ? Self.repetitionPenalty : nil,
            repetitionContextSize: Self.repetitionContext,
            prefill: .init(stepSize: Self.prefillStep)
        )
    }

    private func newSession(
        _ container: ModelContainer, system: String, speculative: SpeculativeDecodingConfig?,
        parameters: GenerateParameters
    ) -> ChatSession {
        ChatSession(
            container, instructions: system, speculativeDecoding: speculative,
            generateParameters: parameters, additionalContext: Self.chatContext
        )
    }

    /// A session whose cache already holds `system`, prefilled once with a
    /// one-token reply; nil if that fails (the caller then starts from scratch).
    private func prefixSession(
        system: String, container: ModelContainer, speculative: SpeculativeDecodingConfig?
    ) async -> (session: ChatSession, reused: Bool)? {
        let key = "\(speculative != nil)\u{1F}\(system)"
        if let index = prefixes.firstIndex(where: { $0.key == key }) {
            let entry = prefixes.remove(at: index)
            prefixes.append(entry)  // most recently used last
            return (entry.session, true)
        }
        let base = newSession(
            container, system: system, speculative: speculative,
            parameters: makeParameters(maxTokens: 1, temperature: 0, penalizeRepeats: false)
        )
        let started = Date()
        do {
            _ = try await base.respond(to: "OK")
        } catch {
            SessionLog.shared.log("IA", "caché del prompt no disponible: \(error.localizedDescription)")
            return nil
        }
        prefixes.append((key, base))
        if prefixes.count > Self.maxCachedPrefixes { prefixes.removeFirst() }
        SessionLog.shared.log(
            "IA", "caché del prompt: variante nueva (\(system.count) car de instrucciones) preparada en "
                + SessionLog.seconds(Date().timeIntervalSince(started))
        )
        return (base, false)
    }

    /// One answer: `system` + `user` → text without reasoning or code fences.
    /// `partial` receives the text so far while it is written (~10 times a second).
    func generate(
        system: String, user: String, maxTokens: Int, temperature: Float = 0.3, penalizeRepeats: Bool = true,
        partial: PartialText? = nil
    ) async throws -> String {
        await acquire()
        defer { release() }
        let container = try await load()
        defer { Memory.clearCache() }
        if !safeMode {
            do {
                let speculative = await draftModel().map {
                    SpeculativeDecodingConfig(draftModel: $0, numDraftTokens: Self.numDraftTokens)
                }
                let params = makeParameters(maxTokens: maxTokens, temperature: temperature, penalizeRepeats: penalizeRepeats)
                // One request = system prompt + one user message, as in Python.
                // The system prompt comes from the prefilled cache when there is one.
                let session: ChatSession
                let prefix: String
                if let prefixed = await prefixSession(system: system, container: container, speculative: speculative) {
                    session = await prefixed.session.fork(history: [])
                    session.generateParameters = params
                    prefix = prefixed.reused ? "reutilizado" : "nuevo"
                } else {
                    session = newSession(container, system: system, speculative: speculative, parameters: params)
                    prefix = "no"
                }
                let mode = "ayudante \(speculative == nil ? "no" : "sí") · prefijo \(prefix) · KV 8 bits desde \(Self.quantizedKVStart)"
                return try await stream(session, user: user, container: container, mode: mode, partial: partial)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // Some optimization does not work here: answer the plain way.
                safeMode = true
                prefixes = []
                draft = nil
                SessionLog.shared.log(
                    "ERROR", "una optimización falló (\(error.localizedDescription)): modo normal para este modelo"
                )
            }
        }
        let plain = newSession(
            container, system: system, speculative: nil,
            parameters: makeParameters(
                maxTokens: maxTokens, temperature: temperature, penalizeRepeats: penalizeRepeats, quantizeKV: false
            )
        )
        return try await stream(plain, user: user, container: container, mode: "modo normal", partial: partial)
    }

    private func stream(
        _ session: ChatSession, user: String, container: ModelContainer, mode: String, partial: PartialText?
    ) async throws -> String {
        let started = ContinuousClock.now
        var firstWord: Duration?
        var output = ""
        var lastUpdate = started
        for try await chunk in session.streamResponse(to: user, role: .user) {
            if firstWord == nil, !chunk.isEmpty { firstWord = ContinuousClock.now - started }
            output += chunk
            if let partial, ContinuousClock.now - lastUpdate >= Self.partialInterval {
                lastUpdate = .now
                partial(AIText.stripCodeFence(AIText.stripThinking(output)))
            }
        }
        let total = Self.seconds(ContinuousClock.now - started)
        let result = AIText.stripCodeFence(AIText.stripThinking(output))
        partial?(result)
        let tokens = await container.encode(output).count
        let writing = max(total - Self.seconds(firstWord ?? .zero), 0.001)
        SessionLog.shared.log(
            "IA", String(format: "%ld tokens en %@ (%.0f tok/s escribiendo) · 1.ª palabra %@ · prompt %ld car · %@ · "
                + "MLX %ld MB (pico %ld MB)",
                tokens, SessionLog.seconds(total), Double(tokens) / writing,
                SessionLog.seconds(Self.seconds(firstWord ?? .zero)), user.count, mode,
                Self.megabytes(Memory.activeMemory), Self.megabytes(Memory.peakMemory))
        )
        return result
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    private static func megabytes(_ bytes: Int) -> Int {
        bytes / 1_048_576
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
