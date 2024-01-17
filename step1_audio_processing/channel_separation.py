import torchaudio
import os
from tqdm import tqdm


def separa_canais(PATH_SRC: str, PATH_DST: str) -> None:
    """
    Essa função separa os canais de áudio e salva em mp3, organizando em pastas com o nome do arquivo e os canais salvos.
    :param PATH_SRC: a pasta onde estão os áudios stereos
    :param PATH_DST: a pasta onde serão salvos os subdiretórios com os canais separados
    :return: None
    """

    if not os.path.exists(PATH_DST):
        os.makedirs(PATH_DST)

    audios = os.listdir(PATH_SRC)
    audios = [audio for audio in audios if audio.endswith('.mp3')]

    ja_foram = []
    for audio in tqdm(audios):
        name_audio = audio.split('.mp3')[0]
        if name_audio in ja_foram:
            continue
        ja_foram.append(name_audio)
        # separando os canais de áudio e salvando novamente em mp3
        wav, sr = torchaudio.load(os.path.join(PATH_SRC, audio))
        assert wav.shape[0] == 2, f"O áudio {audio} não é stereo"
        audio_cliente = wav[:1, :]  # idx 0
        audio_atendente = wav[1:, :]  # idx 1
        del wav

        # salvando os audios
        if not os.path.exists(os.path.join(PATH_DST, name_audio)):
            os.mkdir(os.path.join(PATH_DST, name_audio))  # criando uma pasta para o áudio ser salvo
        torchaudio.save(os.path.join(PATH_DST, name_audio, name_audio + "_cliente_cut.mp3"), audio_cliente, sample_rate=sr)
        torchaudio.save(os.path.join(PATH_DST, name_audio, name_audio + "_atendente_cut.mp3"), audio_atendente, sample_rate=sr)
        # deletando os audios separados
        os.remove(os.path.join(PATH_SRC, audio))
