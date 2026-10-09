def clean_transcript(text):
    """Remove Whisper's non-speech markers, retaining raw output in JSON."""
    import re
    return re.sub(r"\[(?:blank_audio|silence|music|no_speech)\]", "", text,
                  flags=re.IGNORECASE).strip()
