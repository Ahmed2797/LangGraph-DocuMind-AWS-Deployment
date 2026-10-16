import whisper


def load_whisper_model():
    """
    Load the local Whisper model.
    """
    return whisper.load_model("base")

