"""Slim audio file loader for inference (no training-repo dependencies)."""


def load_audio(path, sr=16000):
    """Load an audio file as a 16 kHz mono float32 numpy array.

    soundfile handles wav/flac natively; librosa (audioread/ffmpeg) covers
    mp3/m4a and resampling.
    """
    import librosa
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav
