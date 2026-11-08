import Foundation

/// On-disk memory of a meeting for the AI (port of `meeting_memory.py`, same
/// file format, so both engines share it).
///
/// The transcript is stored in fragments (~3000 characters) in
/// `<directory>/<session>.jsonl`. When asking, only the relevant fragments come
/// back, with BM25 (keyword search, no other model in memory), so the prompt
/// has a fixed size even if the meeting lasts hours.
struct MeetingMemoryStore {
    struct Segment: Codable, Equatable {
        let index: Int
        let start: String
        let end: String
        let text: String
    }

    let url: URL

    enum Failure: LocalizedError {
        case invalidSession(String)
        case missingDirectory

        var errorDescription: String? {
            switch self {
            case .invalidSession(let session): return "Sesión de memoria no válida: \(session)"
            case .missingDirectory: return "Falta la carpeta de la memoria de la reunión."
            }
        }
    }

    init(directory: String, session: String) throws {
        guard !session.isEmpty, session.count <= 64,
              session.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" || $0 == "." || $0 == "-" })
        else { throw Failure.invalidSession(session) }
        guard !directory.isEmpty else { throw Failure.missingDirectory }
        let folder = URL(fileURLWithPath: directory, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        url = folder.appendingPathComponent("\(session).jsonl")
    }

    /// Stored fragments in order; the first one with each index wins (a line
    /// cut by a power loss is skipped).
    func segments() -> [Segment] {
        guard let raw = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        var found: [Int: Segment] = [:]
        let decoder = JSONDecoder()
        for line in raw.split(separator: "\n") {
            guard let data = line.data(using: .utf8), let segment = try? decoder.decode(Segment.self, from: data)
            else { continue }
            if found[segment.index] == nil { found[segment.index] = segment }
        }
        return found.keys.sorted().compactMap { found[$0] }
    }

    /// Stores fragment `index`. Idempotent: repeating it does not duplicate it.
    @discardableResult
    func add(index: Int, start: String, end: String, text: String) throws -> Bool {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !segments().contains(where: { $0.index == index }) else { return false }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        var line = try encoder.encode(Segment(index: index, start: start, end: end, text: text))
        line.append(0x0A)
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } else {
            try line.write(to: url)
        }
        return true
    }

    /// The `k` most relevant fragments for `query` (BM25), in chronological order.
    func search(_ query: String, k: Int = 3, excluding exclude: (Segment) -> Bool = { _ in false }) -> [Segment] {
        let queryTerms = Set(Self.tokenize(query))
        guard !queryTerms.isEmpty else { return [] }
        let candidates = segments().filter { !exclude($0) }
        guard !candidates.isEmpty else { return [] }
        let docs = candidates.map { Self.tokenize($0.text) }
        let averageLength = max(Double(docs.reduce(0) { $0 + $1.count }) / Double(docs.count), 1)
        let count = Double(docs.count)
        var docFrequency: [String: Double] = [:]
        for term in queryTerms {
            docFrequency[term] = Double(docs.filter { $0.contains(term) }.count)
        }
        var scored: [(score: Double, segment: Segment)] = []
        for (segment, words) in zip(candidates, docs) where !words.isEmpty {
            var score = 0.0
            for term in queryTerms {
                let tf = Double(words.filter { $0 == term }.count)
                guard tf > 0, let df = docFrequency[term] else { continue }
                let idf = log(1 + (count - df + 0.5) / (df + 0.5))
                let norm = 1 - Self.b + Self.b * Double(words.count) / averageLength
                score += idf * tf * (Self.k1 + 1) / (tf + Self.k1 * norm)
            }
            if score > 0 { scored.append((score, segment)) }
        }
        let best = scored.sorted { $0.score > $1.score }.prefix(k)
        return best.map(\.segment).sorted { $0.index < $1.index }
    }

    // MARK: Words

    static let k1 = 1.5
    static let b = 0.75

    /// Very frequent words (EN/ES, without accents) that help find nothing.
    static let stopwords: Set<String> = Set("""
        a about after again all also am an and any are as at be because been before being but by can
        could did do does doing for from had has have having he her here hers him his how i if in into
        is it its just me more most my no not now of on once only or other our out over own same she
        should so some such than that the their them then there these they this those through to too
        under until up very was we were what when where which while who whom why will with would you
        your yours yeah okay ok um uh like going gonna get got know think right
        al algo como con cual cuales cuando de del desde donde el ella ellas ellos en entre era eran es
        esa esas ese eso esos esta estaba estamos estan estar este esto estos fue fueron ha han hay la
        las le les lo los mas me mi mis muy nada ni no nos nosotros o para pero por porque que quien se
        sea ser si sin sobre son su sus tambien te tiene tienen todo todos tu tus un una uno unos y ya yo
        bueno pues entonces vale
        """.split(whereSeparator: \.isWhitespace).map(String.init))

    /// Lowercase words without accents, minus stopwords and 1-letter words
    /// ("decisión" finds "decision").
    static func tokenize(_ text: String) -> [String] {
        let folded = text.lowercased().folding(options: [.diacriticInsensitive, .widthInsensitive], locale: nil)
        return folded.split(whereSeparator: { !($0.isLetter || $0.isNumber || $0 == "_") })
            .map(String.init)
            .filter { $0.count > 1 && !stopwords.contains($0) }
    }
}
