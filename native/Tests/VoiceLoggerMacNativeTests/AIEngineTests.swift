@testable import VoiceLoggerMacNative
import XCTest

/// Text logic of the native AI engine (no model needed): the same cases as
/// tests/test_core.py for summarizer.py, meeting_memory.py and web_search.py.
final class AIEngineTests: XCTestCase {
    // MARK: Model output

    func testStripThinking() {
        XCTAssertEqual(AIText.stripThinking("<think>hmm\nok</think>\n## Summary\nX"), "## Summary\nX")
        XCTAssertEqual(AIText.stripThinking("razono...</think>Respuesta"), "Respuesta")
        XCTAssertEqual(AIText.stripThinking("Respuesta<think>cortado"), "Respuesta")
        XCTAssertEqual(AIText.stripCodeFence("```markdown\n## Summary\nOk\n```"), "## Summary\nOk")
    }

    func testCleanMinutesKeepsTheFinalNotes() throws {
        let md = "Okay, so let me draft.\n## Summary\ndraft\nWait, better:\n"
            + "## Summary\nCooking show about snakehead fish noodles.\n## Key questions\n- Where to grab the fish?"
        XCTAssertEqual(
            try AIText.cleanMinutes(md),
            "## Summary\nCooking show about snakehead fish noodles.\n## Key questions\n- Where to grab the fish?"
        )
        XCTAssertThrowsError(try AIText.cleanMinutes("The user wants me to summarize.\n## Summary\nI need to think"))
        XCTAssertThrowsError(try AIText.cleanMinutes("no headings at all"))
    }

    func testCleanAnswer() throws {
        XCTAssertEqual(try AIText.cleanAnswer("Budget approved."), "Budget approved.")
        XCTAssertEqual(try AIText.cleanAnswer("The user asks about X. Let me think.\nAnswer: It was approved."), "It was approved.")
        XCTAssertThrowsError(try AIText.cleanAnswer("The user wants me to answer. Hmm."))
    }

    func testLoopIsCollapsed() throws {
        let loop = "# Resumen\n- **Tema**: Sandwich\n- **Tareas**:\n  - Slicing pork\n"
            + String(repeating: "  - Toasting whole pork shoulder\n", count: 80) + "  -"
        let out = try AIText.cleanMinutes(loop)
        XCTAssertEqual(out.components(separatedBy: "Toasting whole pork shoulder").count - 1, 1)
        XCTAssertEqual(AIText.lines(out).last, "  - Toasting whole pork shoulder")
    }

    func testDedupeKeepsShortLabelsAndOtherSections() {
        let text = "## Option A\n**Pros:**\n- fast\n**Cons:**\n- pricey\n**Pros:**\n- more\n---\n"
            + "## Key questions\n- Who pays for the new office?\n## Summary\n- Who pays for the new office?"
        XCTAssertEqual(AIText.dedupeLines(text), text)
        XCTAssertEqual(AIText.dedupeLines("## Notes\n- **Budget** is approved.\n* Budget is approved\n1. budget IS approved!"),
                       "## Notes\n- **Budget** is approved.")
        XCTAssertEqual(AIText.dedupeLines("- Pork\n- Pork\n- Pork"), "- Pork")
    }

    func testSplitSections() {
        let md = "## Summary\nQ3 review.\n## Key questions\n- Ship before freeze?\n- none\n"
            + "## Decisions\n- Ship it\n## Action items\n1. Julian: fix copy by Friday"
        let (rest, questions) = AIText.splitKeyQuestions(md)
        let (summary, actions) = AIText.splitActionItems(rest)
        XCTAssertEqual(questions, ["Ship before freeze?"])
        XCTAssertEqual(actions, ["Julian: fix copy by Friday"])
        XCTAssertEqual(summary, "## Summary\nQ3 review.\n## Decisions\n- Ship it")
        XCTAssertEqual(AIText.splitActionItems("## Resumen\nX\n## Tareas / pendientes\n1. Enviar acta").items, ["Enviar acta"])
    }

    func testParseQuestions() {
        let raw = "- Why does your organization exist?\n- none\n1. **What is the budget for Q3**\n"
            + "- ok?\n- Why does your organization exist?\n- The user wants questions\n## Notes\n"
            + "- How do we measure ROAS?\n- Who owns the launch?\n- When is the deadline?"
        XCTAssertEqual(AIText.parseQuestions(raw), [
            "Why does your organization exist?", "What is the budget for Q3?",
            "How do we measure ROAS?", "Who owns the launch?",
        ])
        XCTAssertEqual(AIText.parseQuestions("- ninguna"), [])
    }

    func testParseTasks() {
        let raw = "- TASK | OWNER | DUE\n- Send the minutes | Ana | 2026-11-20\n- Book room | - | next week\n"
            + "- Send the minutes | Ana | 2026-11-20\n- Fix bug | nadie | 2026-02-30\n- none"
        let tasks = AIText.parseTasks(raw)
        XCTAssertEqual(tasks.map(\.title), ["Send the minutes", "Book room", "Fix bug"])
        XCTAssertEqual(tasks.map(\.owner), ["Ana", "", ""])
        XCTAssertEqual(tasks.map(\.due), ["2026-11-20", "", ""])  // 30 Feb is not a date
    }

    func testFillAndNothingNew() {
        XCTAssertEqual(AIText.fill("Notes:\n{previous}\nNew:\n{new}", ["previous": "a {new} b", "new": "c"]),
                       "Notes:\na {new} b\nNew:\nc")
        XCTAssertTrue(AIText.saysNothingNew("NADA NUEVO."))
        XCTAssertTrue(AIText.saysNothingNew("Nothing new"))
        XCTAssertFalse(AIText.saysNothingNew("There is nothing new in the meeting, but [1] adds that…" + String(repeating: " x", count: 40)))
    }

    func testSplitAndTailText() {
        let text = (0..<50).map { "[10:\($0)] linea numero \($0)" }.joined(separator: "\n")
        let chunks = AIText.splitText(text, maxChars: 200)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 200 })
        XCTAssertEqual(chunks.joined(separator: "\n"), text)
        let tail = AIText.tailText(text, maxChars: 100, prompts: .es)
        XCTAssertTrue(tail.hasPrefix(AIPrompts.es.trimmed + "\n[10:"))
        XCTAssertTrue(tail.hasSuffix("linea numero 49"))
    }

    func testPromptsMatchBothLanguages() {
        XCTAssertTrue(AIPrompts.en.sections.contains("copy it word for word"))
        XCTAssertTrue(AIPrompts.es.sections.contains("cópiala tal cual"))
        XCTAssertTrue(AIText.systemPrompt(.en, context: "").hasPrefix("You are a meeting assistant. Your job"))
        XCTAssertTrue(AIText.systemPrompt(.es, context: "Eres experto en ventas.", open: true)
            .hasPrefix("Eres experto en ventas.\n\nTu trabajo es ayudarme durante una reunión de lluvia de ideas."))
    }

    // MARK: Meeting memory

    func testMeetingMemoryStoresOnceAndFindsRelevantFragments() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path
        let memory = try MeetingMemoryStore(directory: directory, session: "2026-11-08_10-00-00")
        XCTAssertTrue(try memory.add(index: 0, start: "10:00", end: "10:05", text: "Hablamos del presupuesto de marketing."))
        XCTAssertFalse(try memory.add(index: 0, start: "10:00", end: "10:05", text: "otra vez"))
        XCTAssertTrue(try memory.add(index: 1, start: "10:05", end: "10:10", text: "La decisión: lanzar en enero."))
        XCTAssertTrue(try memory.add(index: 2, start: "10:10", end: "10:15", text: "Clima y fútbol."))
        XCTAssertEqual(memory.segments().count, 3)
        XCTAssertEqual(memory.search("¿Qué decision se tomó sobre el lanzamiento?", k: 1).map(\.index), [1])
        XCTAssertEqual(memory.search("presupuesto", k: 3) { $0.index == 0 }, [])
        XCTAssertThrowsError(try MeetingMemoryStore(directory: directory, session: "../escape"))
    }

    // MARK: Web search

    static let ddgSample = """
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title"><a rel="nofollow" class="result__a"
             href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Findex%2Fwhisper%2F&amp;rut=abc">Introducing <b>Whisper</b></a></h2>
          <a class="result__snippet" href="#">Whisper is a <b>speech recognition</b> model &amp; more.</a>
        </div>
        <div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>
        <div class="result"><h2><a class="result__a" href="https://github.com/openai/whisper">openai/whisper</a></h2>
          <div class="result__snippet">Robust Speech Recognition via Large-Scale Weak Supervision</div></div>
        """

    func testParseDuckDuckGo() {
        let results = WebSearch.parseDuckDuckGo(Self.ddgSample)
        XCTAssertEqual(results.map(\.url), ["https://openai.com/index/whisper/", "https://github.com/openai/whisper"])
        XCTAssertEqual(results.first?.title, "Introducing Whisper")
        XCTAssertEqual(results.first?.snippet, "Whisper is a speech recognition model & more.")
        XCTAssertEqual(results.last?.snippet, "Robust Speech Recognition via Large-Scale Weak Supervision")
    }

    func testParseWikipediaKeepsRanking() throws {
        let json = """
            {"query": {"pages": {
              "2": {"index": 2, "title": "Speech recognition", "fullurl": "https://en.wikipedia.org/wiki/Speech_recognition", "extract": "Speech recognition is a field."},
              "1": {"index": 1, "title": "Whisper (speech recognition system)", "fullurl": "https://en.wikipedia.org/wiki/Whisper", "extract": "Whisper is a machine learning model."}
            }}}
            """
        let results = WebSearch.parseWikipedia(Data(json.utf8))
        XCTAssertEqual(results.map(\.title), ["Whisper (speech recognition system)", "Speech recognition"])
    }

    func testSitesTopicsAndTrust() {
        XCTAssertEqual(WebSearch.parseSites("hbr.org, https://www.McKinsey.com/insights site:who.int hbr.org nada x.org"),
                       ["hbr.org", "mckinsey.com", "who.int"])
        XCTAssertEqual(WebSearch.topicSites(for: "How do I memoize a React component with hooks?"), ["developer.mozilla.org"])
        XCTAssertEqual(WebSearch.topicSites(for: "Cómo mido el ROAS de una campaña con GA4"), ["thinkwithgoogle.com"])
        XCTAssertEqual(WebSearch.topicSites(for: "What are the next steps? Let's rest and lead"), [])
        for url in ["https://en.wikipedia.org/wiki/X", "https://www.nasa.gov/a", "https://www.gov.uk/y",
                    "https://ox.ac.uk/a", "https://react.dev/learn", "https://stackoverflow.com/q/1"] {
            XCTAssertTrue(WebSearch.isTrusted(url), url)
        }
        for url in ["https://randomblog.com/x", "https://abc.go.com/show", "https://fakewikipedia.org/x", ""] {
            XCTAssertFalse(WebSearch.isTrusted(url), url)
        }
        let queries = WebSearch.buildQueries("team motivation", sites: ["hbr.org", "mckinsey.com"])
        XCTAssertEqual(queries.preferred, ["team motivation site:hbr.org", "team motivation site:mckinsey.com"])
        XCTAssertTrue(queries.general.hasPrefix("team motivation -site:pinterest.com"))
    }

    func testRankPreferredThenTrustedThenRest() {
        let urls = ["https://blog.example.com/a", "https://www.pinterest.com/pin/1", "https://en.wikipedia.org/wiki/A",
                    "https://hbr.org/2020/x", "https://blog.example.com/a/", "https://other.com/b"]
        let ranked = WebSearch.rank(urls.map { WebResult(title: $0, url: $0, snippet: "") }, sites: ["hbr.org"])
        XCTAssertEqual(ranked.map(\.url), ["https://hbr.org/2020/x", "https://en.wikipedia.org/wiki/A",
                                           "https://blog.example.com/a", "https://other.com/b"])
    }

    func testPageTextSkipsMenusScriptsAndFooters() {
        let page = """
            <html><head><script>var tracking = "<p>whisper whisper whisper whisper</p>";</script>
            <style>p { color: red }</style></head><body>
            <nav><ul><li>Home</li><li>About Whisper and every other product we sell today online</li></ul></nav>
            <header><p>Subscribe to our newsletter for weekly updates on many different topics</p></header>
            <article>
            <p>Our company was founded many years ago in a small garage with three people.</p>
            <p>Whisper is a speech recognition model trained on 680,000 hours of multilingual audio data.</p>
            <p>It handles accents, background noise and technical language better than older systems.</p>
            <p>Short.</p><br/>
            </article>
            <footer><p>Copyright notice and the cookie policy for this website and its partners</p></footer>
            </body></html>
            """
        let text = WebSearch.extractRelevantText(page, query: "What is Whisper speech recognition?")
        XCTAssertTrue(text.contains("Whisper is a speech recognition model"))
        for absent in ["tracking", "newsletter", "cookie", "every other product", "Short."] {
            XCTAssertFalse(text.contains(absent), absent)
        }
        let small = WebSearch.extractRelevantText(page, query: "speech recognition multilingual audio", maxChars: 100)
        XCTAssertTrue(small.hasPrefix("Whisper is a speech recognition model"))
        XCTAssertLessThanOrEqual(small.count, 101)
    }

    func testOnlyPublicHTTPSPagesAreRead() {
        XCTAssertTrue(WebSearch.isPublicHTTPS("https://example.com/a"))
        for url in ["http://example.com", "https://10.0.0.1/", "https://[::1]/", "https://printer.local/", "file:///etc"] {
            XCTAssertFalse(WebSearch.isPublicHTTPS(url), url)
        }
        XCTAssertFalse(WebSearch.isReadable("https://www.youtube.com/watch?v=1"))
        XCTAssertFalse(WebSearch.isReadable("https://site.com/doc.pdf"))
        XCTAssertTrue(WebSearch.isReadable("https://good.com/page"))
        XCTAssertEqual(WebSearch.unescape("Tom &amp; Jerry &#39;99&#x21; &hellip; &bogus;"), "Tom & Jerry '99! … &bogus;")
    }

    func testWebBlockMarksReliableSourcesAndUsesPageText() {
        let block = AIText.webBlock([
            WebResult(title: "Wiki", url: "https://en.wikipedia.org/wiki/W", snippet: "short", trusted: true,
                      content: "Full paragraph from the page."),
            WebResult(title: "Blog", url: "https://blog.com/x", snippet: "Blog snippet"),
        ], prompts: .en)
        XCTAssertTrue(block.contains("[1] Wiki (reliable source) — https://en.wikipedia.org/wiki/W\nFull paragraph from the page."))
        XCTAssertTrue(block.contains("[2] Blog — https://blog.com/x\nBlog snippet"))
        let sources = AIText.sources([WebResult(title: "Wiki", url: "u", snippet: "", trusted: true, content: "x")])
        XCTAssertEqual(sources, [WebSource(title: "Wiki", url: "u", trusted: true, read: true)])
    }

    // MARK: Direct developer sources

    func testDirectSourcesByTopic() {
        XCTAssertEqual(WebSearch.DirectSource(rawValue: "developer.mozilla.org"), .mdn)
        XCTAssertEqual(WebSearch.DirectSource(rawValue: "stackoverflow.com"), .stackOverflow)
        XCTAssertNil(WebSearch.DirectSource(rawValue: "thinkwithgoogle.com"))  // still a site: search
        XCTAssertEqual(WebSearch.keywords("¿Cómo uso useEffect en React?"), "uso useeffect react")
    }

    func testParseMDN() {
        let json = """
            {"documents": [
              {"mdn_url": "/en-US/docs/Web/API/Fetch_API", "title": "Fetch API",
               "summary": "The Fetch API provides an interface for fetching resources &amp; more.", "score": 10},
              {"mdn_url": "", "title": "Broken"}
            ]}
            """
        let results = WebSearch.parseMDN(Data(json.utf8))
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.title, "MDN: Fetch API")
        XCTAssertEqual(results.first?.url, "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API")
        XCTAssertEqual(results.first?.snippet, "The Fetch API provides an interface for fetching resources & more.")
    }

    func testParseStackOverflowQuestionsAndAcceptedAnswers() {
        let questions = """
            {"items": [
              {"title": "How to use useEffect &amp; cleanup?", "link": "https://stackoverflow.com/q/1", "score": 42,
               "is_answered": true, "accepted_answer_id": 11, "body": "<p>I want to <b>clean up</b>.</p>"},
              {"title": "Unanswered", "link": "https://stackoverflow.com/q/2", "is_answered": false}
            ]}
            """
        let found = WebSearch.parseStackOverflowQuestions(Data(questions.utf8))
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.result.title, "Stack Overflow: How to use useEffect & cleanup? (42 votos)")
        XCTAssertEqual(found.first?.result.snippet, "I want to clean up.")
        XCTAssertEqual(found.first?.accepted, 11)

        let answers = """
            {"items": [{"answer_id": 11, "body": "<p>Return a cleanup function from useEffect so React runs it before the component unmounts or the effect runs again.</p><pre><code>useEffect(() =&gt; { return () =&gt; {} })</code></pre>"}]}
            """
        let texts = WebSearch.parseStackOverflowAnswers(Data(answers.utf8), query: "useEffect cleanup react")
        XCTAssertTrue(texts[11]?.hasPrefix("Respuesta aceptada:\nReturn a cleanup function from useEffect") ?? false)
    }

    // MARK: Tavily and page selection

    func testParseTavilyKeepsScoreAndTheRelevantPageText() {
        let json = """
            {"query": "deploy kubernetes cluster aws", "results": [
              {"title": "Deploy a Kubernetes cluster on AWS", "url": "https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html",
               "content": "Create an Amazon EKS cluster.", "score": 0.91,
               "raw_content": "Menu\\nSign in\\nTo deploy a Kubernetes cluster on AWS, create an EKS cluster with eksctl and choose the node groups that fit your workload.\\nCookie preferences"},
              {"title": "No text", "url": "https://example.com/k8s", "content": "Short summary", "score": 0.4, "raw_content": null},
              {"title": "Bad", "url": "ftp://example.com/file", "content": "x"}
            ]}
            """
        let results = WebSearch.parseTavily(Data(json.utf8), query: "deploy kubernetes cluster aws", limit: 5)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].score, 0.91)
        XCTAssertTrue(results[0].trusted)
        XCTAssertEqual(results[0].snippet, "Create an Amazon EKS cluster.")
        XCTAssertEqual(results[0].content, "To deploy a Kubernetes cluster on AWS, create an EKS cluster with eksctl and choose the node groups that fit your workload.")
        XCTAssertNil(results[1].content)
        XCTAssertEqual(results[1].score, 0.4)
    }

    private func page(_ url: String, _ content: String?, trusted: Bool = false) -> WebResult {
        WebResult(title: url, url: url, snippet: "snippet", trusted: trusted, content: content)
    }

    func testOneGoodPageIsEnough() {
        let results = [
            page("https://a.com", "kubernetes cluster basics"),
            page("https://b.com", "deploy a kubernetes cluster on aws step by step"),
            page("https://c.com", nil),
        ]
        let chosen = WebSearch.selectPages(results, query: "deploy kubernetes cluster aws")
        XCTAssertNil(chosen[0].content)
        XCTAssertNotNil(chosen[1].content)
        XCTAssertEqual(chosen[2].snippet, "snippet")
        XCTAssertEqual(chosen.map(\.url), results.map(\.url))  // order and snippets kept
    }

    func testASecondPageOnlyWhenTheFirstFallsShort() {
        let results = [
            page("https://a.com", "deploy kubernetes"),
            page("https://b.com", "cluster sizing"),
            page("https://c.com", "aws pricing"),
        ]
        let chosen = WebSearch.selectPages(results, query: "deploy kubernetes cluster aws")
        XCTAssertEqual(chosen.filter { $0.content != nil }.map(\.url), ["https://a.com", "https://b.com"])
        // Never more than maxPages, and a single page with text is left alone.
        XCTAssertEqual(WebSearch.selectPages(results, query: "deploy kubernetes cluster aws", maxPages: 1)
            .filter { $0.content != nil }.count, 1)
        XCTAssertEqual(WebSearch.selectPages([results[0]], query: "deploy kubernetes cluster aws"), [results[0]])
    }

    func testReliableSourceWinsATie() {
        let results = [
            page("https://blog.example.com", "deploy kubernetes cluster aws"),
            page("https://docs.aws.amazon.com/eks", "deploy kubernetes cluster aws", trusted: true),
        ]
        let chosen = WebSearch.selectPages(results, query: "deploy kubernetes cluster aws")
        XCTAssertEqual(chosen.filter { $0.content != nil }.map(\.url), ["https://docs.aws.amazon.com/eks"])
    }
}
