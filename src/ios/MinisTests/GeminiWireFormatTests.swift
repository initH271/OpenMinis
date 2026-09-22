import XCTest

/// [T-gemini-empty-part-oneof-400] Regression tests pinning the Gemini
/// wire-format invariant: a part's text is NEVER the empty string, so the
/// protobuf `oneof` is always initialised and the API doesn't 400 with
/// "required oneof field 'data' must have one initialized field". Repro'd on
/// Gemini 3.5 Flash tool calls (empty assistant/model text after a
/// thinking-heavy or safety-stripped turn).
final class GeminiWireFormatTests: XCTestCase {

    // MARK: - nonEmptyText

    func testNonEmptyText_emptyString_becomesSpace() {
        XCTAssertEqual(GeminiWireFormat.nonEmptyText(""), " ")
    }

    func testNonEmptyText_nil_becomesSpace() {
        XCTAssertEqual(GeminiWireFormat.nonEmptyText(nil), " ")
    }

    func testNonEmptyText_nonEmpty_preserved() {
        XCTAssertEqual(GeminiWireFormat.nonEmptyText("hello"), "hello")
    }

    func testNonEmptyText_whitespaceOnly_preserved() {
        // A single space is already valid — don't collapse or trim it away.
        XCTAssertEqual(GeminiWireFormat.nonEmptyText(" "), " ")
        XCTAssertEqual(GeminiWireFormat.nonEmptyText("\n"), "\n")
    }

    // MARK: - textPart (empty text turn → never {"text":""})

    func testTextPart_emptyText_neverEmpty() {
        let part = GeminiWireFormat.textPart("")
        XCTAssertEqual(part["text"] as? String, " ",
                       "{\"text\":\"\"} triggers Gemini oneof 400 — must be substituted")
    }

    func testTextPart_nilText_neverEmpty() {
        XCTAssertEqual(GeminiWireFormat.textPart(nil)["text"] as? String, " ")
    }

    func testTextPart_realText_passesThrough() {
        XCTAssertEqual(GeminiWireFormat.textPart("[Called foo with: {}]")["text"] as? String,
                       "[Called foo with: {}]")
    }

    // MARK: - functionResponseResult (empty tool result content → never empty)

    func testFunctionResponseResult_emptyContent_neverEmpty() {
        let resp = GeminiWireFormat.functionResponseResult("")
        XCTAssertEqual(resp["result"] as? String, " ",
                       "empty tool result would ship an empty string into the response payload")
    }

    func testFunctionResponseResult_nilContent_neverEmpty() {
        XCTAssertEqual(GeminiWireFormat.functionResponseResult(nil)["result"] as? String, " ")
    }

    func testFunctionResponseResult_realContent_passesThrough() {
        XCTAssertEqual(GeminiWireFormat.functionResponseResult("ok")["result"] as? String, "ok")
    }

    // MARK: - Guarantee: no produced part ever carries an empty string

    func testInvariant_noEmptyStringEverProduced() {
        for input: String? in [nil, "", " ", "x", "多字节文本"] {
            XCTAssertFalse((GeminiWireFormat.textPart(input)["text"] as? String)?.isEmpty ?? true,
                           "textPart must never yield an empty string (input=\(String(describing: input)))")
            XCTAssertFalse((GeminiWireFormat.functionResponseResult(input)["result"] as? String)?.isEmpty ?? true,
                           "functionResponseResult must never yield an empty string (input=\(String(describing: input)))")
        }
    }

    // MARK: - usageMetadata → LLMUsage ([T-gemini-cache-usage])

    private func parse(_ metadata: [String: Any]) -> LLMUsage? {
        GeminiWireFormat.usage(from: metadata)
    }

    /// Warm call: `promptTokenCount` is the FULL input (cached included), so the
    /// cached portion must be subtracted to keep `inputTokens` fresh-only — the fix
    /// for "cache rows never appear for Gemini". Real fixture from a warm
    /// gemini-3.8-flash call.
    func testUsage_warmCall_subtractsCacheFromPrompt() {
        let u = parse([
            "promptTokenCount": 45709, "candidatesTokenCount": 48,
            "cachedContentTokenCount": 40926, "thoughtsTokenCount": 1148,
            "totalTokenCount": 46905,
        ])
        XCTAssertEqual(u?.inputTokens, 45709 - 40926, "inputTokens must stay fresh-only")
        XCTAssertEqual(u?.cacheReadInputTokens, 40926)
        XCTAssertEqual(u?.outputTokens, 48 + 1148, "3.x reports thinking outside candidatesTokenCount")
        XCTAssertNil(u?.cacheCreationInputTokens, "Gemini never reports cache writes")
    }

    /// Cold call: the cache key is ABSENT (not 0) — a miss must stay a miss.
    func testUsage_coldCall_cacheReadIsNil() {
        let u = parse([
            "promptTokenCount": 45698, "thoughtsTokenCount": 12, "totalTokenCount": 45710,
        ])
        XCTAssertEqual(u?.inputTokens, 45698)
        XCTAssertNil(u?.cacheReadInputTokens)
    }

    /// A present-but-zero cache count is not a hit either.
    func testUsage_zeroCacheCount_isNotAHit() {
        let u = parse(["promptTokenCount": 500, "candidatesTokenCount": 5, "cachedContentTokenCount": 0])
        XCTAssertNil(u?.cacheReadInputTokens)
        XCTAssertEqual(u?.inputTokens, 500)
    }

    /// Implicit caching can cover only part of the prefix (observed 8179 / 12262 /
    /// 16359 / 40926) — only the reported portion may be subtracted.
    func testUsage_partialPrefixHit_subtractsReportedPortion() {
        let u = parse(["promptTokenCount": 17898, "candidatesTokenCount": 1, "cachedContentTokenCount": 12262])
        XCTAssertEqual(u?.inputTokens, 5636)
        XCTAssertEqual(u?.cacheReadInputTokens, 12262)
    }

    /// 2.5-and-earlier fold thinking INTO `candidatesTokenCount`; the
    /// `totalTokenCount` identity is what keeps them from being counted twice.
    func testUsage_thoughtsFoldedIntoCandidates_notDoubleCounted() {
        let u = parse([
            "promptTokenCount": 46, "candidatesTokenCount": 830,
            "thoughtsTokenCount": 766, "totalTokenCount": 876,
        ])
        XCTAssertEqual(u?.outputTokens, 830, "total proves thoughts are already inside candidates")
    }

    /// No `totalTokenCount` → keep the historical candidates-only reading.
    func testUsage_missingTotal_keepsCandidatesOnly() {
        let u = parse(["promptTokenCount": 100, "candidatesTokenCount": 40, "thoughtsTokenCount": 300])
        XCTAssertEqual(u?.outputTokens, 40)
    }

    /// Progressive chunks can carry a metadata object with no counts at all
    /// (Vertex emits a `trafficType`-only preamble) — it must not be reported as a
    /// zeroed measurement that wipes the turn's cache/context numbers.
    func testUsage_metadataWithoutTokenCounts_isIgnored() {
        XCTAssertNil(parse(["trafficType": "ON_DEMAND"]))
        XCTAssertNil(parse([:]))
    }

    /// Defensive: if the API ever reports more cached tokens than prompt tokens we
    /// keep the FULL prompt (the same fallback the OpenAI/DeepSeek paths use) rather
    /// than inventing a zero.
    func testUsage_cacheLargerThanPrompt_fallsBackToFullPrompt() {
        let u = parse(["promptTokenCount": 500, "candidatesTokenCount": 10, "cachedContentTokenCount": 600])
        XCTAssertEqual(u?.inputTokens, 500)
        XCTAssertEqual(u?.cacheReadInputTokens, 600)
    }
}
