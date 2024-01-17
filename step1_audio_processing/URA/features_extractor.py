import concurrent.futures
import os
import librosa
import numpy as np
import time
import pandas as pd
# desativando warnings
import warnings

warnings.filterwarnings("ignore")

MAX_THREADS = 4


class FeaturesClassicas:
    def __init__(self, TIME_SEGMENT: int = 5, path_audios="", path_save_features: str = "./features", audios_list: list = None):
        """
        Essa classe é responsável por extrair as features clássicas de áudio.
        :param TIME_SEGMENT: tamanho do segmento de áudio em segundos
        :param path_save_features: caminho para salvar as features
        """
        assert TIME_SEGMENT > 0, "O tamanho do segmento deve ser maior que 0"
        assert path_save_features != "", "O caminho para salvar as features não pode ser vazio"
        assert path_audios != "", "O caminho para os áudios não pode ser vazio"

        self.TIME_SEGMENT = TIME_SEGMENT
        self.path_save = path_save_features
        if not os.path.exists(path_save_features):
            os.makedirs(path_save_features)
        self.path_audios = path_audios
        self.path_save = path_save_features
        self.features = []
        # definindo as features de antemão, mas pode ser alterado em tempo de execução
        self.define_features()
        self.audios_list = audios_list

    def run(self, audio_name):
        #print(f"Extraindo features do áudio {audio_name}")
        wave, sr = self.load_file(audio_name)
        segmentos = self.separa_em_segmentos(wave, sr)
        features = self.extrai_features(segmentos, sr)
        df = self.convert_features_to_df(features)
        self.save_features_to_csv(audio_name, df)

    def load_file(self, audio) -> (np.ndarray, int):
        """
        Carrega os arquivos de áudio.
        Os áudios são armazenados no dicionário self.audios
        :param audio: nome do áudio
        :return: None
        """
        # percorrendo cada áudio e carregando
        y, sr = librosa.load(os.path.join(self.path_audios, audio), sr=None, mono=False)
        wave = y[1, :]  # carregando somente do ATENDENTE
        return wave, sr

    def separa_em_segmentos(self, wave, sr) -> np.ndarray:
        """
        Separa os áudios em segmentos de tamanho TIME_SEGMENT.
        Os áudios são armazenados no diciário self.audios
        :param wave: áudio
        :param sr: taxa de amostragem
        :return: None
        """
        segmentacao = librosa.util.frame(wave, frame_length=self.TIME_SEGMENT * sr,
                                         hop_length=self.TIME_SEGMENT * sr)
        return segmentacao

    def define_features(self, features: list = None) -> None:
        """
        Define as features que serão extraídas
        :return: None
        """
        if not features:
            self.features = ["segmento",
                             "mean_centroid",
                             "sd_centroid",
                             "median_centroid",
                             "max_centroid",
                             "min_centroid",
                             "mean_sfm",
                             "median_sfm",
                             "sd_sfm",
                             "mean_rolloff_099",
                             "median_rolloff_099",
                             "sd_rolloff_099",
                             "min_rolloff_099",
                             "max_rolloff_099",
                             "mean_rolloff_001",
                             "median_rolloff_001",
                             "sd_rolloff_001",
                             "min_rolloff_001",
                             "max_rolloff_001",]
                             # "mean_f0",
                             # "median_f0",
                             # "sd_f0",
                             # "min_f0",
                             # "max_f0"]
        else:
            self.features = features

    def extrai_features(self, segmentos: np.ndarray, sr: int) -> list:
        """
        Extrai as features de cada segmento de áudio usando threads.
        :param segmentos: segmentos de áudio
        :param sr: taxa de amostragem
        :return: lista com as features
        """
        data_parallel = []  # lista para armazenar os dados de forma paralela (objeto mutável)
        start = time.time()
        # percorrendo cada segmento de áudio
        for idx_ in range(segmentos.shape[1]):
            segmento = segmentos[:, idx_]
            try:
                extrair_features_audio(segmento, sr, data_parallel, idx_)
            except Exception as e:
                pass

        #print(f"Demorou {time.time() - start:.2f} segundos para extrair features")
        # armazenando os dados das features no dicionario
        return data_parallel

    def convert_features_to_df(self, features) -> pd.DataFrame:
        """
        Converte os dados das features para um dataframe
        :param features: lista com as features
        :return: pd.DataFrame com as features
        """
        # transformando em dataframe
        df_ = pd.DataFrame(features, columns=self.features)
        # ordenando pela coluna segmento
        df_ = df_.sort_values(by=["segmento"])
        # dropando a coluna segmento
        df_ = df_.drop(columns=["segmento"])
        return df_

    def save_features_to_csv(self, audio_name, features, TSV: bool = False) -> None:
        """
        Salva as features em um arquivo CSV ou TSV
        :return: None
        """
        if TSV:
            sep = "\t"
            final = ".tsv"
            header = False
        else:
            sep = ","
            final = ".csv"
            header = self.features[1:]  # removendo a coluna segmento
        # salvando em csv os dados
        features.to_csv(f"{self.path_save}/features_{audio_name.split('/')[-1]}{final}", sep=sep,
                        index=False, header=header)


def extrair_features_audio(audio: np.ndarray, srate: int, data_parallel: list, segmento: int = None):
    """
    Extrai as features de um segmento de áudio
    :param audio: segmento de áudio
    :param srate: taxa de amostragem
    :param data_parallel: lista para armazenar as features de forma assíncrona
    :param segmento: número do segmento para organização dos dados
    :return: list | None
    """
    if data_parallel is not None:
        assert segmento is not None, "É necessário informar o segmento para usar de forma assíncrona"
    # Compute spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=srate)

    # FEATURE 1: Spectral Centroid
    start = time.time()
    # 1.1. Compute mean frequency
    meanfreq = np.mean(centroid)
    # 1.2. Compute standard deviation of frequency
    sd = np.std(centroid)
    # 1.3 Compute Median Frequency
    median = np.median(centroid)
    # 1.4 Compute max and min
    maxfreq = np.max(centroid)
    minfreq = np.min(centroid)
    #print(f"F1 demorou {time.time() - start:.2f} segundos")

    # FEATURE 2: Spectral Flatness
    # Compute spectral flatness
    start = time.time()
    sfm = librosa.feature.spectral_flatness(y=audio)
    # Podemos obter a média, mediana, desvio padrão do spectral flatness
    mean_sfm = np.mean(sfm)
    median_sfm = np.median(sfm)
    sd_sfm = np.std(sfm)
    #print(f"F2 demorou {time.time() - start:.2f} segundos")

    # FEATURE 3: Spectral Rolloff
    # Compute spectral rolloff
    start = time.time()
    spec_rolloff_099 = librosa.feature.spectral_rolloff(y=audio, sr=srate, roll_percent=0.99)
    spec_rolloff_001 = librosa.feature.spectral_rolloff(y=audio, sr=srate, roll_percent=0.01)
    # podemos obter a média, mediana, desvio padrão, minimo e máximo do spectral rolloff
    mean_rolloff_099 = np.mean(spec_rolloff_099)
    median_rolloff_099 = np.median(spec_rolloff_099)
    sd_rolloff_099 = np.std(spec_rolloff_099)
    min_rolloff_099 = np.min(spec_rolloff_099)
    max_rolloff_099 = np.max(spec_rolloff_099)
    # O outro extremo: 0.01
    mean_rolloff_001 = np.mean(spec_rolloff_001)
    median_rolloff_001 = np.median(spec_rolloff_001)
    sd_rolloff_001 = np.std(spec_rolloff_001)
    min_rolloff_001 = np.min(spec_rolloff_001)
    max_rolloff_001 = np.max(spec_rolloff_001)
    #print(f"F3 demorou {time.time() - start:.2f} segundos")

    # FEATURE 4: Fundamental Frequency - PITCH
    # Compute fundamental frequency (pitch)
    start = time.time()
    # f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'),
    #                                              fmax=librosa.note_to_hz('C7'))
    # # Podemos obter a média, mediana, desvio padrão, minimo e máximo do fundamental frequency
    # # ignorar ou remover nan
    # f0 = f0[~np.isnan(f0)]
    # mean_f0 = np.mean(f0)
    # median_f0 = np.median(f0)
    # sd_f0 = np.std(f0)
    # min_f0 = np.min(f0)
    # max_f0 = np.max(f0)
    #print(f"F4 demorou {time.time() - start:.2f} segundos")
    if data_parallel is not None:
        data_parallel.append(
            [segmento, meanfreq, sd, median, maxfreq, minfreq, mean_sfm, median_sfm, sd_sfm, mean_rolloff_099,
             median_rolloff_099, sd_rolloff_099, min_rolloff_099, max_rolloff_099, mean_rolloff_001,
             median_rolloff_001,
             sd_rolloff_001, min_rolloff_001, max_rolloff_001,])# mean_f0, median_f0, sd_f0, min_f0, max_f0])
    else:
        return [meanfreq, sd, median, maxfreq, minfreq, mean_sfm, median_sfm, sd_sfm, mean_rolloff_099,
                median_rolloff_099, sd_rolloff_099, min_rolloff_099, max_rolloff_099, mean_rolloff_001,
                median_rolloff_001, sd_rolloff_001, min_rolloff_001, max_rolloff_001,]
                # mean_f0, median_f0, sd_f0,min_f0, max_f0]


def FC_parallel(audio: str, path_audios: str, path_save_features: str, audios_list: list = None):
    """
    Função para executar a extração de features clássicas de forma paralela.
    :param audio: nome do áudio
    :param path_audios: caminho para os áudios
    :param path_save_features: caminho para salvar as features
    :param audios_list: opcional: lista de áudios
    """

    fc_obj = FeaturesClassicas(
        path_save_features=path_save_features,
        path_audios=path_audios,
        audios_list=audios_list
    )  # instanciando o objeto de extrair features clássicas
    fc_obj.run(audio)


if __name__ == '__main__':
    path_ = "/dados/bronze/anatel"
    path_save_features_ = "/dados/silver/anatel/features_audios"
    audios_list_ = os.listdir(path_)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
        # executando as threads
        for audio in audios_list_:
            executor.submit(FC_parallel, audio, path_, path_save_features_)