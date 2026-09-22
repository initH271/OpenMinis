import Foundation

/// [T-gemini-empty-part-oneof-400] Pure, testable helpers that enforce Gemini's
/// wire-format invariants when building `contents[].parts[]`.
///
/// Gemini's REST API models a part's payload as a protobuf `oneof` ("data"). An
/// EMPTY text part — `{"text": ""}` — is treated as an *uninitialized* oneof and
/// rejected:
///
///   contents[N].parts[M].data: required oneof field 'data' must have one
///   initialized field  (HTTP 400 INVALID_ARGUMENT)
///
/// This bites Gemini 3.x / 3.5 Flash especially: a thinking-heavy turn (whole
/// budget spent thinking) or a safety-stripped candidate can yield an empty
/// assistant text, and a naive `{"text": text}` then ships `{"text": ""}` → 400,
/// surfacing to users as "tool format errors". The fix (cc-plugins #99, verified
/// against the live gateway: `{"text":""}`→400, `{"text":" "}`→200) is to never
/// emit an empty text part — substitute a single space.
///
/// Centralised here so every call site shares one guarantee and a regression
/// test can pin the behaviour.
enum GeminiWireFormat {

    /// Placeholder used in place of an empty text so the `oneof` is always
    /// initialised. A single space is the minimal non-empty, semantically-inert
    /// value (mirrors the Anthropic path's "never send an empty text block").
    static let nonEmptyPlaceholder = " "

    /// Coerce a text value so it is never the empty string. `nil`/`""` → `" "`.
    static func nonEmptyText(_ text: String?) -> String {
        guard let text, !text.isEmpty else { return nonEmptyPlaceholder }
        return text
    }

    /// Build a text part guaranteed to satisfy the oneof constraint (never empty).
    static func textPart(_ text: String?) -> [String: Any] {
        ["text": nonEmptyText(text)]
    }

    /// The `response` object for a `functionResponse` part, guaranteeing the
    /// wrapped `result` string is never empty (an empty tool result would ship an
    /// empty string into the response payload).
    static func functionResponseResult(_ content: String?) -> [String: Any] {
        ["result": nonEmptyText(content)]
    }

    // MARK: - usageMetadata → LLMUsage

    /// Parse Gemini's `usageMetadata` into `LLMUsage`.
    ///
    /// Every rule below is pinned by live traffic (gemini-3.5/3.6/3.8-flash,
    /// official `generativelanguage` surface) plus the Vertex sample in the PR
    /// discussion — see `GeminiWireFormatTests` for the fixtures.
    ///
    /// 1. `promptTokenCount` is the **FULL** input: a warm call reported the same
    ///    50898 as the cold one while also reporting
    ///    `cachedContentTokenCount: 40930`. Cached tokens are therefore *not*
    ///    deducted by the API, and we subtract locally so `inputTokens` keeps its
    ///    fresh-only meaning (same convention as OpenAI/DeepSeek/Anthropic).
    ///    Without the subtraction the cached tokens are counted twice in
    ///    `input + cacheRead`, deflating the cache-hit rate.
    /// 2. `cachedContentTokenCount` is **absent** on a miss (not 0) and can cover
    ///    any subset of the prompt (observed 8179 / 12262 / 16359 / 40926), so it
    ///    is surfaced only when > 0 — which is also what hides the row in
    ///    `UsageStatsView` for non-caching models.
    /// 3. On 3.x, `totalTokenCount == prompt + candidates + thoughts`: thinking is
    ///    NOT folded into `candidatesTokenCount`, so reporting candidates alone
    ///    drops the entire reasoning budget (observed 30 reported vs 766 thoughts
    ///    in one turn). 2.5-and-earlier fold thoughts INTO candidates, so adding
    ///    unconditionally would double-count there — the `totalTokenCount`
    ///    identity is what distinguishes the two conventions, and when it is
    ///    missing we keep the historical candidates-only reading.
    /// 4. Streaming emits **progressive** usage chunks (2–4 per stream, with the
    ///    cache fields only in the final one). A metadata object carrying no token
    ///    counts at all (e.g. Vertex's `trafficType`-only preamble) is dropped
    ///    rather than reported as a zeroed measurement.
    static func usage(from usageMetadata: [String: Any]) -> LLMUsage? {
        let prompt = usageMetadata["promptTokenCount"] as? Int
        let candidates = usageMetadata["candidatesTokenCount"] as? Int
        let cached = usageMetadata["cachedContentTokenCount"] as? Int
        let thoughts = usageMetadata["thoughtsTokenCount"] as? Int

        // Rule 4: ignore partially-populated chunks.
        guard prompt != nil || candidates != nil || cached != nil else { return nil }

        let promptCount = prompt ?? 0
        let candidateCount = candidates ?? 0

        let cacheRead: Int? = (cached ?? 0) > 0 ? cached : nil
        // On a nonsensical payload (cached > prompt) fall back to the FULL prompt, the
        // same guard the OpenAI/DeepSeek paths use (`$0 >= 0 ? $0 : nil` → unchanged),
        // so both platforms and every provider degrade identically.
        let input = (cacheRead.map { promptCount - $0 }).flatMap { $0 >= 0 ? $0 : nil } ?? promptCount

        let thoughtCount = thoughts ?? 0
        let total = usageMetadata["totalTokenCount"] as? Int
        let thoughtsAreSeparate = thoughtCount > 0
            && (total.map { $0 >= promptCount + candidateCount + thoughtCount } ?? false)

        return LLMUsage(
            inputTokens: input,
            outputTokens: thoughtsAreSeparate ? candidateCount + thoughtCount : candidateCount,
            cacheCreationInputTokens: nil,
            cacheReadInputTokens: cacheRead
        )
    }
}
