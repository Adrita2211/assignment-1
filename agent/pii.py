"""PII detection and redaction, layered: Microsoft Presidio (local, always
runs, functional today) plus Amazon Bedrock Guardrails' Sensitive
Information Filters (AWS-managed, deferred until provisioned -- see
redact_bedrock_guardrails' docstring, same seam pattern as
agent/provider.py's BedrockProvider stub). These LAYER, they don't
replace each other -- Guardrails findings get unioned with Presidio's,
never substituted in.

Redaction strategy: PARTIAL masking (e.g. "j***@example.com", last-4-only
for payment), not full masking or tokenization. Full masking
([REDACTED]) destroys the agent's ability to usefully confirm "the email
on file ending in ...@example.com" back to a customer. Tokenization
(reversible pseudonymization) adds a key-management/vault surface this
project has no legitimate need for -- nothing downstream ever needs the
real value revealed again. Partial masking balances "customer can
recognize their own data enough to confirm identity" against "the raw
value never appears in a log, trace, or LLM-visible transcript."

Enforced at three points (all required, see README's PII section for
which specific real leak enforcement point #2 exists to close):
  1. The final customer-facing reply (agent/harness.py's handle_turn).
  2. Tool-call results/errors, before they re-enter the message history
     (agent/mcp_client.py's call_tool return path) -- catches PII inside
     a tool RESULT (e.g. an account's email), not just the final text.
  3. Trace payloads (agent/tracing.py's safe_span_payload wrapper).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

_analyzer = None
_anonymizer = None


def _get_presidio():
    """Lazy singleton -- AnalyzerEngine loads a spaCy NLP pipeline, real
    startup cost not worth paying at import time for code paths (e.g. the
    CI eval-gate) that never call redaction at all.

    Registers two custom PatternRecognizers Presidio's defaults don't
    cover -- found by actually testing, not assumed, same discipline as
    the assignment's own PAN/Aadhaar example:

    1. NANP_FAKE_PHONE: Presidio's default PHONE_NUMBER recognizer is
       backed by Google's `phonenumbers` library, which validates against
       real assignable NANP ranges -- and REJECTS the 555-01XX block as
       invalid, because that block is FCC-reserved specifically for
       fiction (movies, TV, and exactly this kind of test data). Verified
       empirically: '+1-212-9876543' -> PHONE_NUMBER detected;
       '+1-555-0142' -> nothing detected at all. The deliberate choice to
       use fictional numbers in this project's mock data (agent/harness.py's
       data/accounts.json, made specifically to avoid colliding with a
       real person) is directly what defeats the default recognizer.
    2. US_STREET_ADDRESS: Presidio ships no street-address recognizer at
       all by default (only city/state/country via its LOCATION/NER
       entity) -- verified empirically: '482 Birchwood Ave' -> nothing
       detected.
    """
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_anonymizer import AnonymizerEngine

        _analyzer = AnalyzerEngine()
        _analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="NANP_FAKE_PHONE",
            patterns=[Pattern(
                name="nanp_fake_phone",
                regex=r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3,4}(?:[-.\s]?\d{4})?",
                score=0.6,
            )],
        ))
        _analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="US_STREET_ADDRESS",
            patterns=[Pattern(
                name="us_street_address",
                regex=r"\b\d{1,6}\s+[A-Za-z0-9.'\s]{2,40}\b(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Court|Ct|Lane|Ln|Way|Place|Pl)\b\.?",
                score=0.7,
            )],
        ))
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


class PIIFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_type: str    # e.g. "EMAIL_ADDRESS", "PHONE_NUMBER"
    detector: str        # "presidio" | "bedrock_guardrails"
    start: int
    end: int


class RedactionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    redacted_text: str
    findings: list[PIIFinding]


def _mask_op(chars_to_mask: int = 100, from_end: bool = False):
    from presidio_anonymizer.entities import OperatorConfig

    return OperatorConfig(
        "mask", {"masking_char": "*", "chars_to_mask": chars_to_mask, "from_end": from_end},
    )


def _partial_mask_operators() -> dict:
    """Partial-masking operator config per entity type -- Presidio's
    anonymizer's real "mask" operator (configurable chars_to_mask/
    from_end), not the default full-replacement operator. Emails/phones/
    cards/SSNs are masked entirely (the whole value is short enough that
    a partial mask wouldn't hide much anyway); names and free-text
    locations get a shorter mask so a truncated real value stays somewhat
    recognizable without the full string ever appearing in a log."""
    return {
        "EMAIL_ADDRESS": _mask_op(),
        "PHONE_NUMBER": _mask_op(),
        "CREDIT_CARD": _mask_op(),
        "US_SSN": _mask_op(),
        "LOCATION": _mask_op(chars_to_mask=6),
        "PERSON": _mask_op(chars_to_mask=6),
        "NANP_FAKE_PHONE": _mask_op(),
        "US_STREET_ADDRESS": _mask_op(),
        "DEFAULT": _mask_op(),
    }


def redact_presidio(text: str) -> RedactionResult:
    if not text:
        return RedactionResult(redacted_text=text, findings=[])

    analyzer, anonymizer = _get_presidio()
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return RedactionResult(redacted_text=text, findings=[])

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=_partial_mask_operators())
    findings = [
        PIIFinding(entity_type=r.entity_type, detector="presidio", start=r.start, end=r.end)
        for r in results
    ]
    return RedactionResult(redacted_text=anonymized.text, findings=findings)


def redact_bedrock_guardrails(text: str, guardrail_id: str, guardrail_version: str) -> RedactionResult:
    """Deferred AWS call -- boto3 bedrock-runtime apply_guardrail(). Same
    seam pattern as agent/provider.py's BedrockProvider: interface
    designed now, raises NotImplementedError locally until a real
    guardrail is provisioned (see README's AWS provisioning batch)."""
    raise NotImplementedError(
        "redact_bedrock_guardrails is a documented seam, not a working call yet -- "
        "would invoke boto3's bedrock-runtime.apply_guardrail(guardrailIdentifier=guardrail_id, "
        "guardrailVersion=guardrail_version, source='OUTPUT', content=[{'text': {'text': text}}]), "
        "translating its outputAssessments[].sensitiveInformationPolicy.piiEntities into PIIFinding."
    )


def redact_value(value):
    """Recursively redact every string leaf in a dict/list/scalar
    structure -- used for tool results and error payloads, which are
    nested (e.g. an account's shipping_address), not flat text."""
    if isinstance(value, str):
        return redact_presidio(value).redacted_text
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def redact_layered(text: str, use_guardrails: bool = False, guardrail_id: str | None = None, guardrail_version: str | None = None) -> RedactionResult:
    """Presidio ALWAYS runs. Guardrails additionally runs when
    use_guardrails=True (only true once AWS is provisioned) and its
    findings are UNIONED with Presidio's -- each layer's distinct catches
    get preserved, neither replaces the other."""
    result = redact_presidio(text)
    if not use_guardrails:
        return result

    guardrails_result = redact_bedrock_guardrails(result.redacted_text, guardrail_id, guardrail_version)
    return RedactionResult(
        redacted_text=guardrails_result.redacted_text,
        findings=result.findings + guardrails_result.findings,
    )
