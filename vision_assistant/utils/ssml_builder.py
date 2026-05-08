"""
SSML Builder — Generates Speech Synthesis Markup Language for edge-tts.

Builds SSML strings with prosody control (rate, pitch, volume) and
emphasis for urgency-differentiated announcements. edge-tts uses
Microsoft's Neural TTS which supports full SSML spec.
"""


def generate_ssml(
    text: str,
    voice: str = "en-IN-NeerjaNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    emphasis: bool = False,
) -> str:
    """Generate an SSML string for edge-tts with prosody control.

    Args:
        text: The text to speak.
        voice: Microsoft Neural TTS voice name.
        rate: Speech rate adjustment (e.g. "+20%", "-10%").
        pitch: Pitch adjustment (e.g. "+8Hz", "-5Hz").
        volume: Volume adjustment (e.g. "+20%", "-5%").
        emphasis: If True, wrap text in <emphasis level="strong">.

    Returns:
        Complete SSML XML string ready for edge-tts.
    """
    inner = text
    if emphasis:
        inner = f'<emphasis level="strong">{text}</emphasis>'

    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="en-US">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
        f'{inner}'
        f'</prosody>'
        f'</voice>'
        f'</speak>'
    )


# ── Voice Profiles for Urgency Tiers ───────────────────────────

VOICE_PROFILES = {
    "CRITICAL": {
        "rate": "+20%",
        "pitch": "+8Hz",
        "volume": "+20%",
        "emphasis": True,
    },
    "WARNING": {
        "rate": "+5%",
        "pitch": "+0Hz",
        "volume": "+10%",
        "emphasis": False,
    },
    "INFO": {
        "rate": "-5%",
        "pitch": "-2Hz",
        "volume": "+0%",
        "emphasis": False,
    },
    "CLEAR": {
        "rate": "-10%",
        "pitch": "-5Hz",
        "volume": "-5%",
        "emphasis": False,
    },
}


def build_ssml_for_urgency(
    text: str,
    urgency: str,
    voice: str = "en-IN-NeerjaNeural",
) -> str:
    """Build SSML with the voice profile matching the urgency level.

    Args:
        text: The text to speak.
        urgency: One of 'CRITICAL', 'WARNING', 'INFO', 'CLEAR'.
        voice: Microsoft Neural TTS voice name.

    Returns:
        Complete SSML string with urgency-appropriate prosody.
    """
    profile = VOICE_PROFILES.get(urgency, VOICE_PROFILES["INFO"])
    return generate_ssml(
        text=text,
        voice=voice,
        rate=profile["rate"],
        pitch=profile["pitch"],
        volume=profile["volume"],
        emphasis=profile["emphasis"],
    )
