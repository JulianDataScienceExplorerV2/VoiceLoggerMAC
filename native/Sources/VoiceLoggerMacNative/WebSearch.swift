import Foundation

/// A web search result for the AI prompt.
struct WebResult: Sendable, Equatable {
    var title: String
    var url: String
    var snippet: String
    var trusted = false
    /// Paragraphs of the page closest to the question, if the page was read.
    var content: String?
    /// Relevance given by the search API (Tavily: 0-1), if any.
    var score: Double?
}

/// Minimal web search (no API keys or extra packages): port of `web_search.py`.
///
/// Only the question text (plus `site:` operators) goes out: never the audio
/// or the transcript.
/// 1. DuckDuckGo (HTML version; its links come mostly from Bing) without social
///    sites or login walls (`-site:pinterest.com`…).
/// 2. At the same time, one search per preferred site (`… site:hbr.org`;
///    DuckDuckGo does not support `OR` between `site:`) and, by topic, a
///    reference site: frontend → MDN, backend → Stack Overflow, marketing →
///    Think with Google.
/// 3. Fewer than 2 results (or DuckDuckGo asks for a captcha) → Wikipedia.
/// 4. Order: preferred sites, then reliable sources, then the rest.
/// 5. Optionally the best pages are read: their paragraphs closest to the
///    question, not only the search snippet.
enum WebSearch {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static let userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) VoiceLoggerMac/1.0"
    static let timeout: TimeInterval = 8
    static let maxSnippetChars = 600
    static let candidates = 8
    static let maxSites = 3
    static let perSite = 2
    static let pageTimeout: TimeInterval = 5
    static let pageDeadline: Double = 7
    static let maxPageBytes = 1_500_000
    static let pageExcerptChars = 2500

    static let excludedSites = ["pinterest.com", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com"]
    static let unreadable = ["youtube.com", "youtu.be", "vimeo.com", "spotify.com", "linkedin.com"] + excludedSites

    static let trustedDomains = [
        "wikipedia.org", "britannica.com", "who.int", "un.org", "oecd.org", "worldbank.org", "imf.org",
        "europa.eu", "nature.com", "science.org", "sciencedirect.com", "springer.com", "arxiv.org",
        "thelancet.com", "nejm.org", "bmj.com", "mayoclinic.org", "clevelandclinic.org",
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "economist.com", "ft.com", "nytimes.com",
        "theguardian.com", "elpais.com", "hbr.org", "mckinsey.com", "investopedia.com",
        "developer.apple.com", "support.apple.com", "learn.microsoft.com", "developer.mozilla.org",
        "docs.python.org", "cloud.google.com", "aws.amazon.com",
        // Development (official docs and references)
        "web.dev", "developer.chrome.com", "react.dev", "nextjs.org", "vuejs.org", "angular.dev", "svelte.dev",
        "nodejs.org", "typescriptlang.org", "python.org", "djangoproject.com", "fastapi.tiangolo.com",
        "postgresql.org", "docs.docker.com", "kubernetes.io", "git-scm.com", "w3.org", "owasp.org",
        "stackoverflow.com", "martinfowler.com", "developers.google.com", "developer.android.com",
        "rust-lang.org", "go.dev", "scikit-learn.org", "pandas.pydata.org",
        // Marketing and marketing science
        "thinkwithgoogle.com", "ama.org", "marketingscience.info", "nielsen.com", "moz.com",
        "searchengineland.com", "support.google.com", "bcg.com", "gartner.com", "forrester.com",
        "pewresearch.org", "statista.com",
    ]
    private static let trustedTLDs: Set<String> = ["gov", "edu", "int", "mil"]
    private static let trustedSecondLevel: Set<String> = ["gov", "gob", "edu", "ac", "go", "govt"]

    /// Reference site searched with `site:` by topic (words lowercase and
    /// without accents; only ones that are not everyday language).
    static let topicKeywords: [(site: String, keywords: Set<String>)] = [
        ("developer.mozilla.org", [
            "javascript", "typescript", "react", "nextjs", "vue", "angular", "svelte", "css", "html", "dom",
            "tailwind", "frontend", "webpack", "vite", "navegador", "browser", "accesibilidad", "accessibility",
            "jsx", "tsx", "hooks", "flexbox",
        ]),
        ("stackoverflow.com", [
            "python", "django", "flask", "fastapi", "nodejs", "java", "golang", "rust", "sql", "postgres",
            "postgresql", "mysql", "sqlite", "mongodb", "redis", "api", "apis", "graphql", "docker", "kubernetes",
            "k8s", "aws", "azure", "gcp", "backend", "microservicios", "microservices", "oauth", "jwt", "kafka",
            "linux", "bash", "git", "endpoint", "endpoints", "orm", "serverless", "pandas", "numpy",
        ]),
        ("thinkwithgoogle.com", [
            "marketing", "mercadeo", "campaign", "campaigns", "campana", "campanas", "brand", "branding", "seo",
            "sem", "ctr", "cpc", "cpm", "cpa", "roas", "roi", "attribution", "atribucion", "conversion",
            "conversiones", "conversions", "funnel", "embudo", "segmentation", "segmentacion", "ltv", "cac",
            "churn", "retention", "retencion", "audience", "audiencia", "ads", "anuncios", "ga4", "cohort",
            "cohorte", "uplift", "incrementality", "incrementalidad", "mmm", "pricing", "engagement", "leads",
            "crm", "influencer", "influencers",
        ]),
    ]
    static let maxTopicSites = 2

    // MARK: Domains

    /// Host of a URL, lowercase and without "www.".
    static func hostOf(_ url: String) -> String {
        var host = (URLComponents(string: url)?.host ?? "").lowercased()
        while host.hasSuffix(".") { host.removeLast() }
        return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
    }

    static func matches(_ host: String, _ domain: String) -> Bool {
        host == domain || host.hasSuffix("." + domain)
    }

    /// Wikipedia, government, universities, agencies, science, serious press,
    /// official docs and the developer / marketing references above.
    static func isTrusted(_ url: String) -> Bool {
        let host = hostOf(url)
        guard !host.isEmpty else { return false }
        let labels = host.split(separator: ".").map(String.init)
        guard let last = labels.last else { return false }
        if trustedTLDs.contains(last) { return true }
        if last.count == 2, labels.count >= 2, trustedSecondLevel.contains(labels[labels.count - 2]) {
            // gov.uk / gob.mx are the government itself; edu.co, ac.uk… need something before
            if labels.count >= 3 || ["gov", "gob"].contains(labels[labels.count - 2]) { return true }
        }
        return trustedDomains.contains { matches(host, $0) }
    }

    /// "hbr.org, https://www.mckinsey.com/insights site:who.int" → ["hbr.org", "mckinsey.com", "who.int"].
    static func parseSites(_ text: String, limit: Int = maxSites) -> [String] {
        var sites: [String] = []
        for piece in text.lowercased().split(whereSeparator: { $0.isWhitespace || $0 == "," || $0 == ";" }) {
            var raw = String(piece)
            if raw.hasPrefix("site:") { raw = String(raw.dropFirst(5)) }
            if let scheme = raw.range(of: "://") { raw = String(raw[scheme.upperBound...]) }
            raw = String(raw.split(separator: "/", maxSplits: 1, omittingEmptySubsequences: false).first ?? "")
            raw = raw.trimmingCharacters(in: CharacterSet(charactersIn: "."))
            if raw.hasPrefix("www.") { raw = String(raw.dropFirst(4)) }
            let labels = raw.split(separator: ".", omittingEmptySubsequences: false)
            let valid = labels.count >= 2 && labels.allSatisfy { label in
                !label.isEmpty && label.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }
            }
            if valid, !sites.contains(raw) { sites.append(raw) }
        }
        return Array(sites.prefix(limit))
    }

    /// Reference sites for the topic of the question (at most `maxTopicSites`).
    static func topicSites(for query: String) -> [String] {
        let words = Set(MeetingMemoryStore.tokenize(query))
        return Array(topicKeywords.filter { !words.isDisjoint(with: $0.keywords) }.map(\.site).prefix(maxTopicSites))
    }

    /// ([one "… site:x" query per preferred site], general query without social sites).
    static func buildQueries(_ query: String, sites: [String]) -> (preferred: [String], general: String) {
        let general = query + " " + excludedSites.map { "-site:\($0)" }.joined(separator: " ")
        return (sites.map { "\(query) site:\($0)" }, general)
    }

    // MARK: HTML helpers

    private static func regex(_ pattern: String, _ options: NSRegularExpression.Options = []) -> NSRegularExpression {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: pattern, options: options)
    }

    private static let tagPattern = regex("<[^>]+>")
    private static let anchorPattern = regex("<a\\b([^>]*)>([\\s\\S]*?)</a\\s*>", [.caseInsensitive])
    private static let snippetPattern = regex(
        "<(a|div|td|span)\\b([^>]*\\bresult__snippet\\b[^>]*)>([\\s\\S]*?)</\\1\\s*>", [.caseInsensitive]
    )
    private static let markupPattern = regex(
        "<!--[\\s\\S]*?-->|<(/?)([a-zA-Z][a-zA-Z0-9]*)\\b[^>]*?(/?)>", []
    )

    private static func attribute(_ name: String, in attributes: String) -> String? {
        let pattern = regex("\\b\(name)\\s*=\\s*(?:\"([^\"]*)\"|'([^']*)')", [.caseInsensitive])
        guard let match = pattern.firstMatch(in: attributes, range: NSRange(attributes.startIndex..., in: attributes))
        else { return nil }
        for group in [1, 2] {
            if let range = Range(match.range(at: group), in: attributes) { return String(attributes[range]) }
        }
        return nil
    }

    private static let namedEntities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ", "hellip": "…",
        "ndash": "–", "mdash": "—", "lsquo": "‘", "rsquo": "’", "ldquo": "“", "rdquo": "”", "laquo": "«",
        "raquo": "»", "copy": "©", "reg": "®", "trade": "™", "middot": "·", "bull": "•", "euro": "€",
    ]

    /// `&amp;`, `&#39;`, `&#x27;`… → characters (common entities).
    static func unescape(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var result = ""
        var index = text.startIndex
        while let amp = text[index...].firstIndex(of: "&") {
            result += text[index..<amp]
            guard let semicolon = text[amp...].prefix(12).firstIndex(of: ";") else {
                result.append("&")
                index = text.index(after: amp)
                continue
            }
            let name = String(text[text.index(after: amp)..<semicolon])
            var replacement: String?
            if name.hasPrefix("#x") || name.hasPrefix("#X") {
                replacement = UInt32(name.dropFirst(2), radix: 16).flatMap(Unicode.Scalar.init).map { String($0) }
            } else if name.hasPrefix("#") {
                replacement = UInt32(name.dropFirst()).flatMap(Unicode.Scalar.init).map { String($0) }
            } else {
                replacement = namedEntities[name]
            }
            if let replacement {
                result += replacement
                index = text.index(after: semicolon)
            } else {
                result.append("&")
                index = text.index(after: amp)
            }
        }
        result += text[index...]
        return result
    }

    static func collapse(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// Tags removed, entities decoded, whitespace collapsed.
    static func cleanText(_ html: String) -> String {
        let noTags = tagPattern.stringByReplacingMatches(in: html, range: NSRange(html.startIndex..., in: html), withTemplate: "")
        return collapse(unescape(noTags))
    }

    // MARK: DuckDuckGo / Wikipedia

    /// DuckDuckGo wraps links: //duckduckgo.com/l/?uddg=<real url>.
    static func realURL(_ href: String) -> String {
        var href = href
        if href.hasPrefix("//") { href = "https:" + href }
        guard let components = URLComponents(string: href), let host = components.host,
              host.hasSuffix("duckduckgo.com"), components.path.hasPrefix("/l/")
        else { return href }
        let target = components.queryItems?.first(where: { $0.name == "uddg" })?.value ?? ""
        return target.isEmpty ? href : target
    }

    /// Results of html.duckduckgo.com (a.result__a + .result__snippet), without ads.
    static func parseDuckDuckGo(_ html: String, limit: Int = 4) -> [WebResult] {
        let range = NSRange(html.startIndex..., in: html)
        // Titles with their position; each snippet belongs to the title before it.
        var found: [(position: Int, result: WebResult)] = []
        for match in anchorPattern.matches(in: html, range: range) {
            guard let attributesRange = Range(match.range(at: 1), in: html),
                  let bodyRange = Range(match.range(at: 2), in: html) else { continue }
            let attributes = String(html[attributesRange])
            let classes = (attribute("class", in: attributes) ?? "").split(separator: " ")
            guard classes.contains("result__a") else { continue }
            let url = realURL(unescape(attribute("href", in: attributes) ?? ""))
            let title = cleanText(String(html[bodyRange]))
            found.append((match.range.location, WebResult(title: title, url: url, snippet: "")))
        }
        for match in snippetPattern.matches(in: html, range: range) {
            guard let bodyRange = Range(match.range(at: 3), in: html) else { continue }
            guard let owner = found.lastIndex(where: { $0.position < match.range.location }) else { continue }
            if found[owner].result.snippet.isEmpty {
                found[owner].result.snippet = String(cleanText(String(html[bodyRange])).prefix(maxSnippetChars))
            }
        }
        var results: [WebResult] = []
        for item in found {
            let result = item.result
            if result.title.isEmpty || !result.url.hasPrefix("http") || result.url.contains("duckduckgo.com/y.js") {
                continue  // ads and internal links
            }
            results.append(result)
            if results.count >= limit { break }
        }
        return results
    }

    static func parseWikipedia(_ json: Data, limit: Int = 4) -> [WebResult] {
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let query = root["query"] as? [String: Any] else { return [] }
        var pages: [[String: Any]] = []
        if let dict = query["pages"] as? [String: Any] {
            pages = dict.values.compactMap { $0 as? [String: Any] }
        } else if let list = query["pages"] as? [[String: Any]] {
            pages = list
        }
        pages.sort { ($0["index"] as? Int ?? 0) < ($1["index"] as? Int ?? 0) }
        return pages.prefix(limit).compactMap { page in
            guard let title = page["title"] as? String, !title.isEmpty,
                  let url = page["fullurl"] as? String, !url.isEmpty else { return nil }
            let extract = collapse(page["extract"] as? String ?? "")
            return WebResult(title: title, url: url, snippet: String(extract.prefix(maxSnippetChars)))
        }
    }

    private static func decode(_ data: Data, encodingName: String?) -> String {
        if let encodingName {
            let cf = CFStringConvertIANACharSetNameToEncoding(encodingName as CFString)
            if cf != kCFStringEncodingInvalidId {
                let encoding = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cf))
                if encoding != .utf8, let text = String(data: data, encoding: encoding) { return text }
            }
        }
        return String(decoding: data, as: UTF8.self)
    }

    private static func fetch(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("en,es;q=0.8", forHTTPHeaderField: "Accept-Language")
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw Failure(message: "HTTP \(http.statusCode)")
        }
        return data
    }

    /// When DuckDuckGo fails (timeout, error, "are you a bot?" page), searches
    /// skip it for a while and go straight to Wikipedia, instead of waiting
    /// for the timeout on every question (seen in a session log: 8-11 s each).
    private actor DuckDuckGoPause {
        private var until: Date?

        func isPaused() -> Bool {
            if let until, until > Date() { return true }
            return false
        }

        /// True if this call started the pause (to log it once).
        func pause(for seconds: TimeInterval) -> Bool {
            let wasPaused = isPaused()
            until = Date().addingTimeInterval(seconds)
            return !wasPaused
        }
    }

    private static let duckDuckGoPause = DuckDuckGoPause()
    static let duckDuckGoPauseSeconds: TimeInterval = 120

    private static func duckDuckGo(_ query: String) async throws -> [WebResult] {
        if await duckDuckGoPause.isPaused() {
            throw Failure(message: "DuckDuckGo en pausa tras un fallo reciente")
        }
        var components = URLComponents(string: "https://html.duckduckgo.com/html/")!
        components.queryItems = [URLQueryItem(name: "q", value: query)]
        let html: String
        do {
            html = decode(try await fetch(components.url!), encodingName: nil)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as URLError where error.code == .cancelled {
            throw error  // the search was cancelled on purpose: not DuckDuckGo's fault
        } catch {
            await pauseDuckDuckGo(reason: error.localizedDescription)
            throw error
        }
        let lowered = html.lowercased()
        if lowered.contains("anomaly-modal") || lowered.contains("bots use duckduckgo") {
            await pauseDuckDuckGo(reason: "pidió verificación (anti-bots)")
            throw Failure(message: "DuckDuckGo pidió verificación")
        }
        return parseDuckDuckGo(html, limit: candidates)
    }

    private static func pauseDuckDuckGo(reason: String) async {
        if await duckDuckGoPause.pause(for: duckDuckGoPauseSeconds) {
            SessionLog.shared.log(
                "WEB", "DuckDuckGo falló (\(reason)): \(Int(duckDuckGoPauseSeconds / 60)) min solo con Wikipedia"
            )
        }
    }

    private static func wikipedia(_ query: String, language: String, limit: Int) async throws -> [WebResult] {
        var components = URLComponents(string: "https://\(language == "es" ? "es" : "en").wikipedia.org/w/api.php")!
        components.queryItems = [
            "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
            "gsrlimit": String(limit), "prop": "extracts|info", "exintro": "1", "explaintext": "1",
            "exlimit": String(limit), "inprop": "url",
        ].map { URLQueryItem(name: $0.key, value: $0.value) }
        return parseWikipedia(try await fetch(components.url!), limit: limit)
    }

    // MARK: Direct developer sources (JSON APIs, no captcha, no key)

    /// Developer topics are searched straight in the official APIs instead of
    /// `site:` searches on DuckDuckGo: MDN's search API for frontend, the Stack
    /// Exchange API (questions with their accepted answers) for backend. Like
    /// Wikipedia, they answer in JSON and do not block apps.
    enum DirectSource: String, Sendable {
        case mdn = "developer.mozilla.org"
        case stackOverflow = "stackoverflow.com"
    }

    static let directResults = 3

    /// Words to search for: without stopwords or accents ("¿Cómo uso useEffect en React?" → "uso useeffect react").
    static func keywords(_ query: String) -> String {
        MeetingMemoryStore.tokenize(query).prefix(8).joined(separator: " ")
    }

    static func parseMDN(_ json: Data, limit: Int = directResults) -> [WebResult] {
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let documents = root["documents"] as? [[String: Any]] else { return [] }
        return documents.prefix(limit).compactMap { document in
            guard let title = document["title"] as? String, !title.isEmpty,
                  let path = document["mdn_url"] as? String, !path.isEmpty else { return nil }
            let url = path.hasPrefix("http") ? path : "https://developer.mozilla.org" + path
            let summary = collapse(unescape(document["summary"] as? String ?? ""))
            return WebResult(title: "MDN: \(title)", url: url, snippet: String(summary.prefix(maxSnippetChars)))
        }
    }

    /// Stack Overflow questions: (result, accepted answer id). Questions
    /// without an answer are skipped; the question body is the snippet.
    static func parseStackOverflowQuestions(_ json: Data, limit: Int = directResults) -> [(result: WebResult, accepted: Int?)] {
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let items = root["items"] as? [[String: Any]] else { return [] }
        var found: [(result: WebResult, accepted: Int?)] = []
        for item in items {
            guard let title = item["title"] as? String, let link = item["link"] as? String,
                  (item["is_answered"] as? Bool) == true else { continue }
            let score = item["score"] as? Int ?? 0
            let body = cleanText(item["body"] as? String ?? "")
            found.append((
                WebResult(title: "Stack Overflow: \(unescape(title)) (\(score) votos)", url: link,
                          snippet: String(body.prefix(maxSnippetChars))),
                item["accepted_answer_id"] as? Int
            ))
            if found.count >= limit { break }
        }
        return found
    }

    /// Accepted answers by id → their text (the most relevant part).
    static func parseStackOverflowAnswers(_ json: Data, query: String) -> [Int: String] {
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let items = root["items"] as? [[String: Any]] else { return [:] }
        var answers: [Int: String] = [:]
        for item in items {
            guard let id = item["answer_id"] as? Int, let body = item["body"] as? String else { continue }
            let relevant = extractRelevantText(body, query: query)
            let text = relevant.isEmpty ? String(cleanText(body).prefix(pageExcerptChars)) : relevant
            if !text.isEmpty { answers[id] = "Respuesta aceptada:\n" + text }
        }
        return answers
    }

    private static func mdn(_ query: String) async throws -> [WebResult] {
        var components = URLComponents(string: "https://developer.mozilla.org/api/v1/search")!
        components.queryItems = [URLQueryItem(name: "q", value: query), URLQueryItem(name: "locale", value: "en-US")]
        return parseMDN(try await fetch(components.url!))
    }

    private static func stackOverflow(_ query: String) async throws -> [WebResult] {
        var components = URLComponents(string: "https://api.stackexchange.com/2.3/search/advanced")!
        components.queryItems = [
            "order": "desc", "sort": "relevance", "q": query, "site": "stackoverflow", "answers": "1",
            "pagesize": "6", "filter": "withbody",
        ].map { URLQueryItem(name: $0.key, value: $0.value) }
        var found = parseStackOverflowQuestions(try await fetch(components.url!))
        let ids = found.compactMap { $0.accepted }
        guard !ids.isEmpty else { return found.map { $0.result } }
        var answersURL = URLComponents(
            string: "https://api.stackexchange.com/2.3/answers/" + ids.map(String.init).joined(separator: ";")
        )!
        answersURL.queryItems = [URLQueryItem(name: "site", value: "stackoverflow"), URLQueryItem(name: "filter", value: "withbody")]
        if let data = try? await fetch(answersURL.url!) {
            let answers = parseStackOverflowAnswers(data, query: query)
            for index in found.indices {
                if let id = found[index].accepted, let text = answers[id] { found[index].result.content = text }
            }
        }
        return found.map { $0.result }
    }

    /// The direct sources for these topics, in parallel; a failing one is skipped (and logged).
    private static func directSearch(_ query: String, sources: [DirectSource]) async -> [WebResult] {
        let words = keywords(query)
        guard !sources.isEmpty, !words.isEmpty else { return [] }
        return await withTaskGroup(of: (Int, [WebResult]).self) { group in
            for (slot, source) in sources.enumerated() {
                group.addTask {
                    do {
                        switch source {
                        case .mdn: return (slot, try await mdn(words))
                        case .stackOverflow: return (slot, try await stackOverflow(words))
                        }
                    } catch {
                        SessionLog.shared.log("WEB", "\(source.rawValue) no respondió: \(error.localizedDescription)")
                        return (slot, [])
                    }
                }
            }
            var bySlot: [Int: [WebResult]] = [:]
            for await (slot, found) in group { bySlot[slot] = found }
            return sources.indices.flatMap { bySlot[$0] ?? [] }
        }
    }

    // MARK: Tavily (optional, with the user's key)

    /// Search API made for AI apps: no anti-bot pages and the text of each page
    /// already extracted (one request, 1 credit: 1000 free a month). Used first
    /// when there is a key in ⚙; DuckDuckGo / Wikipedia stay as fallback.
    static func parseTavily(_ json: Data, query: String, limit: Int) -> [WebResult] {
        guard let root = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let items = root["results"] as? [[String: Any]] else { return [] }
        return items.prefix(limit).compactMap { item in
            guard let url = item["url"] as? String, url.hasPrefix("http") else { return nil }
            let title = (item["title"] as? String).map(collapse) ?? hostOf(url)
            let snippet = collapse(item["content"] as? String ?? "")
            var content: String?
            if let raw = item["raw_content"] as? String, !raw.isEmpty {
                let relevant = extractRelevantPlainText(raw, query: query)
                content = relevant.isEmpty ? nil : relevant
            }
            return WebResult(
                title: title.isEmpty ? hostOf(url) : title, url: url,
                snippet: String(snippet.prefix(maxSnippetChars)), trusted: isTrusted(url), content: content,
                score: (item["score"] as? Double) ?? (item["score"] as? NSNumber)?.doubleValue
            )
        }
    }

    /// `withText`: also bring each page's text (same cost), for "leer páginas".
    private static func tavily(_ query: String, key: String, limit: Int, withText: Bool) async throws -> [WebResult] {
        var request = URLRequest(url: URL(string: "https://api.tavily.com/search")!, timeoutInterval: 12)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "query": query, "search_depth": "basic", "max_results": limit,
            "include_raw_content": withText ? ("text" as Any) : (false as Any),
        ] as [String: Any])
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        switch status {
        case 200: return parseTavily(data, query: query, limit: limit)
        case 401, 403: throw Failure(message: "Tavily rechazó la clave (revísala en ⚙)")
        case 429, 432, 433: throw Failure(message: "Tavily: se acabaron las búsquedas del plan o hay demasiadas seguidas")
        default: throw Failure(message: "Tavily HTTP \(status)")
        }
    }

    // MARK: Which pages the AI reads

    /// The AI reads as little as it needs: the best page first (most of the
    /// question's keywords, reliable source, search relevance); a second one
    /// only if the first does not cover the question well (60 % of its
    /// keywords), and a third only if the first two together still do not.
    /// The other pages keep just their snippet. Reading text is what makes the
    /// model slow to start, so fewer, better pages = faster answer.
    static func selectPages(_ results: [WebResult], query: String, maxPages: Int = 3) -> [WebResult] {
        let terms = Set(MeetingMemoryStore.tokenize(query))
        let withText = results.indices.filter { results[$0].content != nil }
        guard !terms.isEmpty, withText.count > 1 else { return results }
        func covered(_ index: Int) -> Set<String> {
            terms.intersection(MeetingMemoryStore.tokenize(results[index].content ?? ""))
        }
        func quality(_ index: Int) -> Double {
            Double(covered(index).count) / Double(terms.count)
                + (results[index].trusted ? 0.15 : 0) + (results[index].score ?? 0) * 0.3
        }
        let order = withText.sorted { quality($0) != quality($1) ? quality($0) > quality($1) : $0 < $1 }
        var kept: [Int] = []
        var coveredTerms = Set<String>()
        for index in order {
            if kept.count >= maxPages { break }
            if !kept.isEmpty, Double(coveredTerms.count) / Double(terms.count) >= 0.6 { break }
            kept.append(index)
            coveredTerms.formUnion(covered(index))
        }
        var trimmed = results
        for index in withText where !kept.contains(index) { trimmed[index].content = nil }
        SessionLog.shared.log(
            "WEB", "la IA lee \(kept.count) de \(withText.count) páginas (cubren el "
                + "\(Int(Double(coveredTerms.count) / Double(terms.count) * 100)) % de la pregunta)"
        )
        return trimmed
    }

    private static func urlKey(_ url: String) -> String {
        var key = String(url.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first ?? "")
        while key.hasSuffix("/") { key.removeLast() }
        return key.replacingOccurrences(of: "://www.", with: "://").lowercased()
    }

    /// Preferred sites first, then reliable sources, then the rest; no repeats
    /// and no social sites. Within each group the engine order is kept.
    static func rank(_ results: [WebResult], sites: [String]) -> [WebResult] {
        var unique: [WebResult] = []
        var seen = Set<String>()
        for result in results {
            let key = urlKey(result.url)
            let host = hostOf(result.url)
            if seen.contains(key) || host.isEmpty || excludedSites.contains(where: { matches(host, $0) }) { continue }
            seen.insert(key)
            unique.append(result)
        }
        func score(_ result: WebResult) -> Int {
            let host = hostOf(result.url)
            if sites.contains(where: { matches(host, $0) }) { return 2 }
            return isTrusted(result.url) ? 1 : 0
        }
        // Stable: sort by (score desc, original position).
        return unique.enumerated().sorted { a, b in
            let (sa, sb) = (score(a.element), score(b.element))
            return sa != sb ? sa > sb : a.offset < b.offset
        }.map(\.element)
    }

    // MARK: Reading pages

    static func isPublicHTTPS(_ url: String) -> Bool {
        guard let components = URLComponents(string: url), components.scheme?.lowercased() == "https",
              let rawHost = components.host?.lowercased(), !rawHost.isEmpty else { return false }
        let host = rawHost.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        if host == "localhost" || host.hasSuffix(".local") || host.hasSuffix(".internal") { return false }
        // IP literals never (keeps the local network out).
        if host.contains(":") { return false }
        if host.split(separator: ".").allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }) { return false }
        return true
    }

    /// Only follows redirects to another public https page.
    private final class SafeRedirects: NSObject, URLSessionTaskDelegate, Sendable {
        func urlSession(
            _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
            newRequest request: URLRequest
        ) async -> URLRequest? {
            guard let url = request.url?.absoluteString, WebSearch.isPublicHTTPS(url) else { return nil }
            return request
        }
    }

    private static func fetchPage(_ url: String) async throws -> String {
        guard let address = URL(string: url) else { return "" }
        var request = URLRequest(url: address, timeoutInterval: pageTimeout)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
        request.setValue("en,es;q=0.8", forHTTPHeaderField: "Accept-Language")
        let (bytes, response) = try await URLSession.shared.bytes(for: request, delegate: SafeRedirects())
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
              ["text/html", "application/xhtml+xml"].contains(response.mimeType ?? "")
        else { return "" }  // PDF, image, download, redirect refused…
        var data = Data()
        data.reserveCapacity(256 * 1024)
        for try await byte in bytes {
            data.append(byte)
            if data.count >= maxPageBytes { break }
        }
        return decode(data, encodingName: response.textEncodingName)
    }

    private static let skipTags: Set<String> = [
        "script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button",
        "select", "template", "iframe",
    ]
    private static let rawTextTags: Set<String> = ["script", "style"]
    private static let blockTags: Set<String> = [
        "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "td", "section", "article",
        "main", "blockquote", "dd", "dt", "pre", "ul", "ol", "table",
    ]

    /// Page text in blocks (paragraphs, items, titles), without menus,
    /// headers, footers, scripts or forms.
    static func textBlocks(_ html: String) -> [String] {
        let ns = html as NSString
        var blocks: [String] = []
        var buffer = ""
        var skip = 0
        var position = 0
        func flush() {
            let text = collapse(unescape(buffer))
            buffer = ""
            if !text.isEmpty { blocks.append(text) }
        }
        while position < ns.length {
            guard let match = markupPattern.firstMatch(in: html, range: NSRange(location: position, length: ns.length - position))
            else {
                if skip == 0 { buffer += ns.substring(from: position) }
                break
            }
            if skip == 0, match.range.location > position {
                buffer += ns.substring(with: NSRange(location: position, length: match.range.location - position))
            }
            position = match.range.location + match.range.length
            let nameRange = match.range(at: 2)
            guard nameRange.location != NSNotFound else { continue }  // comment
            let name = ns.substring(with: nameRange).lowercased()
            let closing = match.range(at: 1).length > 0
            let selfClosing = match.range(at: 3).length > 0
            if !closing, rawTextTags.contains(name) {
                // <script>/<style>: jump to the closing tag (their content is not HTML).
                let end = ns.range(of: "</\(name)", options: .caseInsensitive, range: NSRange(location: position, length: ns.length - position))
                if end.location == NSNotFound { break }
                let close = ns.range(of: ">", range: NSRange(location: end.location, length: ns.length - end.location))
                position = close.location == NSNotFound ? ns.length : close.location + 1
                continue
            }
            if skipTags.contains(name) {
                if selfClosing { continue }
                skip = closing ? max(0, skip - 1) : skip + 1
            } else if blockTags.contains(name) {
                flush()
            }
        }
        flush()
        return blocks
    }

    private static func cut(_ text: String, _ maxChars: Int) -> String {
        guard text.count > maxChars else { return text }
        let head = String(text.prefix(maxChars))
        if let space = head.lastIndex(of: " ") { return String(head[..<space]) + "…" }
        return head + "…"
    }

    /// The page paragraphs closest to `query` (up to `maxChars`), in page
    /// order. Without matches, the beginning.
    static func extractRelevantText(_ html: String, query: String, maxChars: Int = pageExcerptChars) -> String {
        relevantText(from: textBlocks(html), query: query, maxChars: maxChars)
    }

    /// Same for plain text (Tavily's page text): one block per line.
    static func extractRelevantPlainText(_ text: String, query: String, maxChars: Int = pageExcerptChars) -> String {
        let lines = text.split(whereSeparator: \.isNewline).map { collapse(String($0)) }
        return relevantText(from: lines, query: query, maxChars: maxChars)
    }

    private static func relevantText(from candidates: [String], query: String, maxChars: Int) -> String {
        var blocks: [String] = []
        var seen = Set<String>()
        for block in candidates
        where block.count >= 60 && block.split(whereSeparator: \.isWhitespace).count >= 8 && !seen.contains(block) {
            seen.insert(block)
            blocks.append(block)
        }
        guard !blocks.isEmpty else { return "" }
        let terms = Set(MeetingMemoryStore.tokenize(query))
        let hits = blocks.map { terms.intersection(MeetingMemoryStore.tokenize($0)).count }
        let order = blocks.indices.sorted { hits[$0] != hits[$1] ? hits[$0] > hits[$1] : $0 < $1 }
        var chosen: [Int] = []
        var total = 0
        for i in order {
            if total >= maxChars { break }
            let room = maxChars - total
            if blocks[i].count > room, !chosen.isEmpty { continue }  // does not fit whole: maybe a shorter one does
            chosen.append(i)
            total += min(blocks[i].count, room) + 1
        }
        return String(chosen.sorted().map { cut(blocks[$0], maxChars) }.joined(separator: "\n").prefix(maxChars + 1))
    }

    static func isReadable(_ url: String) -> Bool {
        let host = hostOf(url)
        let path = (url.split(separator: "?", maxSplits: 1).first.map(String.init) ?? url).lowercased()
        return isPublicHTTPS(url) && !path.hasSuffix(".pdf") && !unreadable.contains { matches(host, $0) }
    }

    /// Adds `content` to the first `count` readable pages, in parallel. Pages
    /// that fail or take longer than the deadline keep their snippet.
    static func readPages(_ results: [WebResult], query: String, count: Int) async -> [WebResult] {
        var results = results
        let targets = Array(results.indices.filter { results[$0].content == nil && isReadable(results[$0].url) }.prefix(count))
        guard !targets.isEmpty else { return results }
        let urls = targets.map { results[$0].url }
        await withTaskGroup(of: (Int, String?).self) { group in
            for (slot, url) in urls.enumerated() {
                group.addTask {
                    guard let html = try? await fetchPage(url), !html.isEmpty else { return (slot, nil) }
                    let excerpt = extractRelevantText(html, query: query)
                    return (slot, excerpt.isEmpty ? nil : excerpt)
                }
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(pageDeadline))
                return (-1, nil)  // deadline
            }
            var finished = 0
            for await (slot, excerpt) in group {
                if slot < 0 { break }
                if let excerpt { results[targets[slot]].content = excerpt }
                finished += 1
                if finished == urls.count { break }
            }
            group.cancelAll()
        }
        return results
    }

    // MARK: Search

    /// Up to `limit` results. `sites`: preferred domains ("hbr.org, who.int");
    /// `read`: how many of the best pages to open and read (0 = only snippets);
    /// `topics`: add the topic reference site (MDN, Stack Overflow…).
    static func search(
        _ query: String, language: String = "en", limit: Int = 4, sites: String = "", read: Int = 0,
        topics: Bool = true, tavilyKey: String? = nil
    ) async throws -> [WebResult] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }
        if let tavilyKey, !tavilyKey.isEmpty {
            let started = Date()
            do {
                let topicList = topics ? topicSites(for: query) : []
                let direct = topicList.compactMap(DirectSource.init(rawValue:))
                async let fromDirect = directSearch(query, sources: direct)
                let fromTavily = try await tavily(query, key: tavilyKey, limit: max(limit, 5), withText: read > 0)
                SessionLog.shared.log(
                    "WEB", "Tavily: \(fromTavily.count) resultados en \(SessionLog.seconds(Date().timeIntervalSince(started)))"
                )
                // Preferred and topic sites (MDN, Stack Overflow, Think with Google…) go first.
                let found = await fromDirect + fromTavily
                var results = Array(rank(found, sites: parseSites(sites) + topicList).prefix(limit))
                if results.count >= 2 {
                    for index in results.indices { results[index].trusted = isTrusted(results[index].url) }
                    // Tavily already brings the text of its pages: download only what is missing.
                    let missing = read - results.filter { $0.content != nil }.count
                    if missing > 0 { results = await readPages(results, query: query, count: missing) }
                    return selectPages(results, query: query, maxPages: max(read, 1))
                }
            } catch {
                SessionLog.shared.log("WEB", "Tavily no respondió (\(error.localizedDescription)): búsqueda normal")
            }
        }
        var siteList = parseSites(sites)
        var direct: [DirectSource] = []
        if topics {
            for site in topicSites(for: query) {
                if let source = DirectSource(rawValue: site) {
                    direct.append(source)  // its own API instead of a site: search
                } else if siteList.count < maxSites, !siteList.contains(site) {
                    siteList.append(site)
                }
            }
        }
        // Results from the direct sources rank like preferred sites.
        let rankSites = siteList + direct.map(\.rawValue)
        let (preferredQueries, generalQuery) = buildQueries(query, sites: siteList)

        async let directFound = Self.directSearch(query, sources: direct)
        async let generalSearch: Result<[WebResult], Error> = Self.run(generalQuery)
        let preferred = await withTaskGroup(of: (Int, [WebResult]).self) { group in
            for (slot, preferredQuery) in preferredQueries.enumerated() {
                group.addTask { (slot, (try? await duckDuckGo(preferredQuery)) ?? []) }
            }
            var bySlot: [Int: [WebResult]] = [:]
            for await (slot, found) in group { bySlot[slot] = Array(found.prefix(perSite)) }
            return preferredQueries.indices.flatMap { bySlot[$0] ?? [] }
        }
        var general: [WebResult] = []
        var failure: Error?
        switch await generalSearch {
        case .success(let found): general = found
        case .failure(let error): failure = error
        }
        if general.isEmpty, failure == nil {  // the -site: filters may have emptied the search
            switch await run(query) {
            case .success(let found): general = found
            case .failure(let error): failure = error
            }
        }
        var results = rank(await directFound + preferred + general.prefix(candidates), sites: rankSites)

        if results.count < 2 {
            do {
                var found = try await wikipedia(query, language: language, limit: limit)
                // Wikipedia's search often finds nothing for a whole question:
                // try again with its keywords only.
                let keywords = MeetingMemoryStore.tokenize(query).prefix(6).joined(separator: " ")
                if found.isEmpty, !keywords.isEmpty, keywords != query.lowercased() {
                    found = try await wikipedia(keywords, language: language, limit: limit)
                }
                results = rank(results + found, sites: rankSites)
            } catch {
                if results.isEmpty, failure != nil {
                    throw Failure(message: "No pude buscar en internet (\(error.localizedDescription)).")
                }
            }
        }

        results = Array(results.prefix(limit))
        for index in results.indices { results[index].trusted = isTrusted(results[index].url) }
        if read > 0 { results = await readPages(results, query: query, count: read) }
        return selectPages(results, query: query, maxPages: max(read, 1))
    }

    private static func run(_ query: String) async -> Result<[WebResult], Error> {
        do { return .success(try await duckDuckGo(query)) } catch { return .failure(error) }
    }
}
