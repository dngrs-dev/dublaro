from dataclasses import dataclass

from dublaro.adapters.tts import TtsAdapter
from dublaro.schemas import VoiceProfile


@dataclass(frozen=True)
class SpeakerVoice:
    profile: VoiceProfile
    adapter: TtsAdapter
