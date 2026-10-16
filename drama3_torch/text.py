"""Drama 3 prompt rows, text validation, long-form chunking and the expected-length estimate.

Ports of audio.cpp's fish_drama3 prompt builder (which matches fish-speech's
build_drama3_prompt), its tag-aware text chunker, and drama3-small's length guard fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

STRUCTURAL = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:")
SPEAKER = re.compile(r"<\|speaker:([0-9]+)\|>")
# Unicode White_Space, as Python's prompt builder uses (str.strip() alone also strips U+001C..U+001F).
WHITE_SPACE = "".join(map(chr, (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
                                0x2028, 0x2029, 0x202F, 0x205F, 0x3000)))
TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"


@dataclass
class Take:
    """A transcript with its codes [num_codebooks, frames] (a reference or an earlier take)."""

    text: str
    codes: torch.Tensor


def validate(text: str, location: str) -> None:
    if not text.strip(WHITE_SPACE):
        raise ValueError(f"{location}: text is empty")
    for token in STRUCTURAL:
        if token in text:
            raise ValueError(f"{location}: {token} is reserved for the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > 15:
            raise ValueError(f"{location}: speaker tags must be <|speaker:0|> to <|speaker:15|>")


def tagged(text: str, reference: bool) -> str:
    """Prefix <|speaker:0|> unless a speaker tag leads; in references the tag moves before leading space."""
    body = text.lstrip(WHITE_SPACE)
    match = SPEAKER.match(body)
    if match is None:
        return "<|speaker:0|> " + text
    if not reference:
        return text
    leading = text[: len(text) - len(body)]
    return match.group(0) + leading + body[match.end():]


def build_prompt(tokenizer, text: str, references: list[Take], history: list[Take],
                 semantic_begin: int, num_codebooks: int) -> torch.Tensor:
    """Prompt rows [rows, 1 + num_codebooks] (int64, CPU): text rows [token, 0...], audio rows [c0 + begin, c0..cN-1]."""
    rows: list[torch.Tensor] = []

    def text_rows(segment: str) -> None:
        ids = torch.tensor(tokenizer.encode(segment, add_special_tokens=False).ids, dtype=torch.long)
        r = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
        r[:, 0] = ids
        rows.append(r)

    def code_rows(codes: torch.Tensor) -> None:
        codes = codes.to("cpu", torch.long)
        if codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise ValueError(f"codes must be [{num_codebooks}, frames >= 1]")
        rows.append(torch.cat((codes[:1] + semantic_begin, codes)).T)

    def user_turn(value: str) -> None:
        validate(value, "text")
        text_rows("<|im_start|>user\n")
        text_rows(tagged(value, False))
        text_rows("<|im_end|>\n")
        text_rows("<|im_start|>assistant\n<|voice|>")

    text_rows("<|im_start|>system\n")
    text_rows(REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION)
    for index, ref in enumerate(references, 1):
        validate(ref.text, "reference")
        text_rows(f"\n\n[Reference {index}]\nText: {tagged(ref.text, True)}\nSpeech: ")
        code_rows(ref.codes)
    text_rows("<|im_end|>\n")
    for take in history:
        user_turn(take.text)
        code_rows(take.codes)
        text_rows("<|im_end|>\n")
    user_turn(text)
    return torch.cat(rows)


# ---------------------------------------------------------------------------
# Length guard


_CJK_PUNCT = {0x3002, 0x3001, 0xFF01, 0xFF1F, 0x300C, 0x300D, 0x2026, 0x30FB, 0xFF08, 0xFF09}


def expected_frames(text: str) -> float:
    """Expected frames for the spoken text (21.5 frames/s), fitted on clean drama3-small takes:
    3.88 per CJK/kana character + 2.00 per ASCII letter or digit + 23.9 per control span + 8.5.
    A leading "Direction: ...\\nLine:" note is not spoken."""
    spoken = text
    if spoken.startswith("Direction:") and "\nLine:" in spoken:
        spoken = spoken[spoken.index("\nLine:") + 6:]
    cjk = latin = spans = 0
    square = angle = 0
    for ch in spoken:
        cp = ord(ch)
        if ch == "[":
            square += 1
            spans += 1
        elif ch == "]":
            square = max(0, square - 1)
        elif ch == "<":
            angle += 1
            spans += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif square or angle:
            continue
        elif cp >= 0x3000 and cp not in _CJK_PUNCT:
            cjk += 1
        elif cp < 0x80 and ch.isalnum():
            latin += 1
    return 3.88 * cjk + 2.00 * latin + 23.9 * spans + 8.5


# ---------------------------------------------------------------------------
# Tag-aware chunking (audio.cpp engine::text, TagAware mode)

_SENTENCE = {".", "!", "?", "。", "！", "？"}
_ASCII_SPACE = set(" \t\n\r\f\v")


def _units(text: str) -> list[tuple[int, int, bool, bool]]:
    """(start, end, is_tag, ends_sentence) character ranges."""
    out, pos, n = [], 0, len(text)
    while pos < n:
        while pos < n and text[pos] in _ASCII_SPACE:
            pos += 1
        if pos >= n:
            break
        start, tag = pos, False
        if text[pos] in "[<":
            close = "]" if text[pos] == "[" else ">"
            end = text.find(close, pos + 1)
            if end >= 0:
                pos, tag = end + 1, True
            else:
                pos += 1
                while pos < n and text[pos] not in _ASCII_SPACE and text[pos] not in "[<":
                    pos += 1
        else:
            pos += 1
            while pos < n and text[pos] not in _ASCII_SPACE and text[pos] not in "[<" and text[pos - 1] not in _SENTENCE:
                pos += 1
        out.append((start, pos, tag, not tag and text[pos - 1] in _SENTENCE))
    return out


def _chunk_body(text: str, budget: int) -> list[str]:
    if len(text) <= budget:
        return [text]
    units = _units(text)
    if not units:
        return []
    length = lambda a, b: units[b - 1][1] - units[a][0]  # noqa: E731 - codepoints of units a..b-1
    join = lambda a, b: text[units[a][0]:units[b - 1][1]].strip(" \t\n\r\f\v")  # noqa: E731
    sentences, start = [], 0
    for i, u in enumerate(units):
        if u[3]:
            sentences.append((start, i + 1))
            start = i + 1
    if start < len(units):
        sentences.append((start, len(units)))
    chunks, s = [], 0
    while s < len(sentences):
        a, b = sentences[s]
        if length(a, b) > budget:  # one long sentence: split at unit boundaries
            u = a
            while u < b:
                e = u
                while e < b and length(u, e + 1) <= budget:
                    e += 1
                e = max(e, u + 1)
                if piece := join(u, e):
                    chunks.append(piece)
                u = e
            s += 1
            continue
        e = s
        while e < len(sentences) and length(sentences[s][0], sentences[e][1]) <= budget:
            e += 1
        e = max(e, s + 1)
        if piece := join(sentences[s][0], sentences[e - 1][1]):
            chunks.append(piece)
        s = e
    return chunks


def chunk_text(text: str, budget: int = 200) -> list[str]:
    """Split long text at sentence ends (then unit boundaries) into pieces of <= budget characters.
    [tags] and <spans> are never split; a leading "(...)" control is repeated on every chunk."""
    text = text.strip(" \t\n\r\f\v")
    if not text:
        return []
    if text.startswith("(") and ")" in text:
        close = text.index(")")
        prefix, body = text[: close + 1].strip(), text[close + 1:].strip()
        if prefix and body:
            return [f"{prefix} {c}" for c in _chunk_body(body, max(1, budget - len(prefix) - 1))]
    return _chunk_body(text, budget)
