import os
import torchaudio
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
from tools import check_path_exists
from sys import getsizeof
import requests


class Denoising:
    def __init__(self, tmp_files_path: str = './dados/bronze/tmp_files'):
        """
        Essa classe realiza a remoção de ruído de um áudio utilizando diferentes técnicas de denoising.
        :param tmp_files_path:
        """
        self.tmp_files_path = tmp_files_path

        # declarando o modelo de denoising
        self.denoiser_model = None

    @staticmethod
    def denoise_nuvem(audio_path: str = None, save_path: str = None, audio_name: str = None) -> None:
        """
        Essa função realiza a remoção de ruído de um áudio utilizando diferentes técnicas de denoising.
        Salva o áudio denoised no mesmo diretório do áudio original.
        :param save_path: o caminho onde o áudio denoised será salvo.
        :param audio_path: o caminho do arquivo de áudio a ser removido o ruído.
        :param audio_name: o nome do arquivo de áudio a ser removido o ruído.
        :return: None.
        """
        audio_file = open(os.path.join(audio_path, audio_name), "rb")

        url = "http://200.137.197.69:43016/denoiser"

        # Realize a requisição POST com o arquivo de áudio
        files = {"client_speech": ('audio.opus', audio_file)}
        response = requests.post(url, files=files)

        # Verifique a resposta
        if response.status_code == 200:
            save_path = os.path.join(save_path, audio_name.replace('.opus', '.ogg'))
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print("Erro na requisição POST. Código de status:", response.status_code)
            print(response.text)
            exit(-1)

    def denoise(self, audio_path: str = None, save_path: str = None, tensor: torch.tensor = None,
                sample_rate: int = None, audio_name: str = None) -> torch.tensor:
        """
        Essa função realiza a remoção de ruído de um áudio utilizando diferentes técnicas de denoising.
        Salva o áudio denoised no mesmo diretório do áudio original.
        :param save_path: opcional: o caminho onde o áudio denoised será salvo.
        :param audio_path: opcional: o caminho do arquivo de áudio a ser removido o ruído.
        :param tensor: opcional: tensor do áudio, se não for passado o caminho do áudio.
        :param sample_rate: opcional: sample rate do áudio, se não for passado o caminho do áudio.
        :param audio_name: opcional: o nome do arquivo de áudio a ser removido o ruído.
        :return: o tensor do áudio denoised.
        """
        if not self.denoiser_model:
            self.denoiser_model = pretrained.dns64().cuda() if torch.cuda.is_available() else pretrained.dns64()  # TODO: pq esse modelo em especifico?
        # Denoiser usando do facebook
        if audio_path:
            wav, sample_rate = torchaudio.load(audio_path)
            wav = wav.cuda() if torch.cuda.is_available() else wav
        else:
            # verificando se o áudio está no cuda
            if tensor.is_cuda:
                wav = tensor
            else:
                wav = tensor.cuda() if torch.cuda.is_available() else tensor
            if sample_rate != 16000:
                wav = convert_audio(wav, sample_rate, self.denoiser_model.sample_rate, self.denoiser_model.chin)

        denoised_audio = torch.tensor([]).cuda() if torch.cuda.is_available() else torch.tensor([])
        with torch.no_grad():
            # dividindo o audio em trechos de 30 segundos e aplicando o denoiser em cada trecho
            trecho = 120
            for i in range(0, wav.shape[1], sample_rate * trecho):
                denoised_audio = torch.cat((denoised_audio, self.denoiser_model(wav[:, i:i + sample_rate * trecho])[0]),
                                           dim=1)
            del wav

        if save_path:
            # salvando a versão denoised do áudio no mesmo diretório do áudio original, com frequencia de amostragem
            # de 16000
            check_path_exists(save_path)
            save_path = os.path.join(save_path, audio_name)
            torchaudio.save(save_path, denoised_audio.cpu(), 16000, format="mp3")
        else:
            return denoised_audio

    def separate_channels(self, audio_name: str = None, audio_wave: torch.tensor = None, sr: int = None,
                          path_voicerecords: str = None, path_save: str = None):
        """
        Essa função separa os canais de um áudio em dois arquivos diferentes.
        :param audio_name: opcional: o nome do arquivo de áudio a ser separado.
        :param audio_wave: opcional: o tensor do áudio a ser separado.
        :param sr: opcional: o sample rate do áudio a ser separado.
        :param path_voicerecords: opcional: o caminho da pasta onde os áudios serão salvos.
        :param path_save: opcional: o caminho da pasta onde os áudios serão salvos.
        :return: None | tuple(torch.tensor(cliente, atendente)) -> caso seja passado o tensor do áudio, retorna o tensor do áudio separado.
        """
        if not path_voicerecords and audio_name:
            audio_path = audio_name
        elif audio_name:
            audio_path = os.path.join(path_voicerecords, audio_name)
        # Denoiser do facebook
        if audio_name:
            audio_wave, sr = torchaudio.load(audio_path)

        # extraindo os canais do áudio (cliente e atendente)
        audio_cliente = audio_wave[:1, :]  # idx 0
        audio_atendente = audio_wave[1:, :]  # idx 1

        if sr != 16000:
            audio_cliente = torchaudio.transforms.Resample(sr, 16000)(
                audio_cliente)
            audio_atendente = torchaudio.transforms.Resample(sr, 16000)(
                audio_atendente)

        if path_save:
            torchaudio.save(self.tmp_files_path + f'/cliente_aux_{audio_name}', audio_cliente, 16000)
            torchaudio.save(self.tmp_files_path + f'/atendente_aux_{audio_name}', audio_atendente, 16000)
        else:
            return audio_cliente, audio_atendente


def denoisar(audio_name: str = None, dns_object: Denoising = None, path_bronze: str = None, path_silver: str = None, mp=False):
    if mp:
        dns_object = Denoising()
    # split dos canais
    audio_cliente, audio_atendente = dns_object.separate_channels(audio_name=audio_name,
                                                                  path_voicerecords=path_bronze)
    # denoise de cada canal separadamente e salva na pasta silver
    audio_cliente = dns_object.denoise(tensor=audio_cliente, sample_rate=16000)
    audio_atendente = dns_object.denoise(tensor=audio_atendente, sample_rate=16000)

    # unindo e salvando os canais denoised
    audio_denoised = torch.cat((audio_cliente, audio_atendente), dim=0)
    audio_denoised = audio_denoised.cpu()
    del audio_cliente, audio_atendente
    audio_name = audio_name.replace('.opus', '.mp3')
    torchaudio.save(os.path.join(path_silver, audio_name), audio_denoised, 16000, format="mp3")

def denoisar_nuvem(audio_name: str = None, dns_object: Denoising = None, path_bronze: str = None,
                   path_silver: str = None):
    # split dos canais
    audio_cliente, audio_atendente = dns_object.separate_channels(audio_name=audio_name,
                                                                  path_voicerecords=path_bronze)

    # salvando os canais separados em ogg
    save_path_bronze = os.path.join(path_bronze, "tmp")
    if not os.path.exists(save_path_bronze):
        os.makedirs(save_path_bronze)

    save_file = os.path.join(save_path_bronze, audio_name.replace(".opus", ""))
    torchaudio.save(f"{save_file}_cliente.ogg", audio_cliente, 16000, format="ogg")
    torchaudio.save(f"{save_file}_atendente.ogg", audio_atendente, 16000, format="ogg")

    # denoisando de fato
    dns_object.denoise_nuvem(audio_path=save_path_bronze, save_path=path_silver,
                             audio_name=audio_name.replace(".opus", "_cliente.ogg"))
    dns_object.denoise_nuvem(audio_path=save_path_bronze, save_path=path_silver,
                             audio_name=audio_name.replace(".opus", "_atendente.ogg"))


if __name__ == '__main__':
    dns = Denoising()
    path_bronze = "../dados/bronze/anatel"
    path_silver = "../dados/silver/anatel"
    audio = "0a0d8d80-0dd2-4cf2-80bd-63de9f2ea37c.opus"
    denoisar_nuvem(audio_name=audio, dns_object=dns, path_bronze=path_bronze, path_silver=path_silver)
    # limpando o arquivo temporário
    input()
    os.remove(os.path.join(path_bronze, "tmp"))
