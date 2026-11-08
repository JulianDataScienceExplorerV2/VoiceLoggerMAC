import Foundation

/// A web search result for the AI prompt.
struct WebResult: Sendable, Equatable {
    var title: String
    var url: String
    var snippet: String
    var trusted = false
    /// Paragraphs of the page closest to the question, if the page was read.
    var content: String?
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
        var blocks: [String] = []
        var seen = Set<String>()
        for block in textBlocks(html)
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
        let targets = Array(results.indices.filter { isReadable(results[$0].url) }.prefix(count))
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
        topics: Bool = true
    ) async throws -> [WebResult] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }
        var siteList = parseSites(sites)
        if topics {
            for site in topicSites(for: query) where siteList.count < maxSites && !siteList.contains(site) {
                siteList.append(site)
            }
        }
        let (preferredQueries, generalQuery) = buildQueries(query, sites: siteList)

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
        var results = rank(preferred + general.prefix(candidates), sites: siteList)

        if results.count < 2 {
            do {
                var found = try await wikipedia(query, language: language, limit: limit)
                // Wikipedia's search often finds nothing for a whole question:
                // try again with its keywords only.
                let keywords = MeetingMemoryStore.tokenize(query).prefix(6).joined(separator: " ")
                if found.isEmpty, !keywords.isEmpty, keywords != query.lowercased() {
                    found = try await wikipedia(keywords, language: language, limit: limit)
                }
                results = rank(results + found, sites: siteList)
            } catch {
                if results.isEmpty, failure != nil {
                    throw Failure(message: "No pude buscar en internet (\(error.localizedDescription)).")
                }
            }
        }

        results = Array(results.prefix(limit))
        for index in results.indices { results[index].trusted = isTrusted(results[index].url) }
        if read > 0 { results = await readPages(results, query: query, count: read) }
        return results
    }

    private static func run(_ query: String) async -> Result<[WebResult], Error> {
        do { return .success(try await duckDuckGo(query)) } catch { return .failure(error) }
    }
}
