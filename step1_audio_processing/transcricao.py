import torch
from transformers import pipeline
# from step1_audio_processing.utils.temp_audio import create_temp_audio, delete_temp_audio
from faster_whisper import WhisperModel
import os
import requests
from transformers.utils import is_flash_attn_2_available


class STT:
    def __init__(self, path_voicerecords: str, sample_rate: int = 16000):
        self.path_voicerecords = path_voicerecords
        if not os.path.exists(self.path_voicerecords):
            os.makedirs(self.path_voicerecords)

        self.sample_rate = sample_rate

        self.DURACOES_AUDIOS = []
        print('----- IS GPU ENABLED: -----')
        print(torch.cuda.is_available())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.faster_whisper = None
        self.IFwhisper = None

    def get_local_faster_whisper_transcription(self, audio) -> str:
        transcript = ""
        # caso o modelo não tenha sido carregado ainda
        if self.faster_whisper is None:
            self.IFwhisper = None
            self.faster_whisper = WhisperModel("large-v3", device=self.device, compute_type="int8")

        assert ".wav" in audio.filename or ".mp3" in audio.filename, "The file must have an .wav or .mp3 extension"
        segments, _ = self.faster_whisper.transcribe(audio,
                                                     vad_filter=True,
                                                     vad_parameters=dict(min_silence_duration_ms=2200,
                                                                         threshold=0.7,
                                                                         min_speech_duration_ms=250,
                                                                         window_size_samples=1024,
                                                                         speech_pad_ms=400),
                                                     language='pt')
        result = []
        for segment in segments:
            result.append((segment.start, segment.end, segment.text))

        return result

    def get_local_insanely_fast_whisper_transcription(self, audio) -> list:
        assert ".wav" in audio or ".mp3" in audio, "The file must have an .wav or .mp3 extension"

        transcript = ""
        # verificando se o modelo não foi carregado ainda
        if self.IFwhisper is None:
            self.faster_whisper = None
            self.IFwhisper = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
                torch_dtype=torch.float16,
                device="cuda:0",  # or mps for Mac devices
                model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {
                    "attn_implementation": "sdpa"},
            )

        # gerando os outputs
        outputs = self.IFwhisper(audio,
                                 chunk_length_s=30,
                                 batch_size=24,
                                 generate_kwargs={"language": "portuguese"},
                                 return_timestamps=True)
        chunks = outputs['chunks']
        del outputs
        # formatando no formato start, end, transcription
        outputs = [(chunk['timestamp'][0], chunk['timestamp'][1], chunk['text']) for chunk in chunks]
        return outputs

    def get_transcription_nuvem(self, audio_path):
        """
        Essa função recebe o tensor referente a um áudio e retorna a transcrição do mesmo.
        :param audio_path: caminho do áudio
        :return: tupla contendo a transcrição e a duração do áudio.
        """

        audio_file = open(audio_path, "rb")
        url = "http://200.137.197.69:43016/local_faster_whisper_transcription"

        # Realize a requisição POST com o arquivo de áudio
        files = {"client_speech": ('audio.wav', audio_file)}
        response = requests.post(url, files=files)

        # Verifique a resposta
        if response.status_code == 200:
            return response.json()
        else:
            raise ("Erro na requisição POST. Código de status:", response.status_code)
