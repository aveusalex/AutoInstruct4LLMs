# usando Kmeans para clusterizar os segmentos
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf


class RemoveURA:
    def __init__(self, path_csvs: str, csvs: list = None):
        """
        Essa classe é responsável por remover a URA dos áudios.
        :param path_csvs:
        :param csvs: opcional: uma lista com os nomes dos arquivos csvs. Se None, lista todos os csvs do diretório.
        """
        self.kmeans = None
        self.path_csvs = path_csvs
        self.filter = np.array([-1, -1, 1, 1])
        self.csvs = csvs

    def run_default(self, TIME_SEGMENT: int) -> pd.DataFrame:
        self.find_csvs()
        self.initiate_kmeans()
        df_final = pd.DataFrame(columns=["audio", "inicio_atendimento"])
        for csv in self.csvs:
            df = self.load_csv(csv)
            labels = self.clusteriza_segmentos(df)
            point_of_change = self.filtering(labels)
            # printando o ponto de troca e estimando o tempo que a URA acaba
            # print(f"O ponto de troca é: {point_of_change}")
            # print(f"{csv} - O tempo que a URA acaba é: {point_of_change * TIME_SEGMENT} segundos")
            df_aux = pd.DataFrame({"audio": [csv], "inicio_atendimento": [point_of_change * TIME_SEGMENT]})
            # concatenando
            df_final = pd.concat([df_final, df_aux])

        # salvando o resultado
        df_final.to_csv("trocas_URA_atendimento.csv", index=False)
        return df_final

    def find_csvs(self) -> None:
        """
        Essa função busca os arquivos csvs no diretório fornecido.
        :return: None
        """
        if self.csvs is None:
            self.csvs = os.listdir(self.path_csvs)

    def load_csv(self, csv):
        """
        Essa função carrega um csv.
        :return: pd.DataFrame com o csv carregado
        """
        df = pd.read_csv(os.path.join(self.path_csvs, csv), )
        return df

    def initiate_kmeans(self, n_clusters: int = 2, random_state: int = 0, n_init: int = 10):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    def clusteriza_segmentos(self, df: pd.DataFrame) -> np.ndarray:
        """
        Essa função clusteriza os segmentos de áudio com o algoritmo Kmeans.
        Essa abordagem tenta clusterizar os áudios em 2 classes: URA e não URA.
        Faz essa clusterização com base somente nas features do PRÓPRIO áudio, sem levar em consideração os outros.
        :param df: pd.DataFrame com os segmentos de áudio
        :return: np.ndarray com as classes encontradas
        """
        X = df
        kmeans = self.kmeans.fit(X)
        return kmeans.labels_

    def filtering(self, binary_sequence) -> int:
        """
        Essa função filtra a sequência binária fornecida com o filtro definido na classe.
        Basicamente, realiza uma convolução 1D com o filtro e retorna o resultado.
        :param binary_sequence: uma sequência binária com as classes encontradas no áudio
        :param threshold: threshold para encontrar o ponto de troca
        :return: o index do ponto de troca
        """
        # Aplicar a convolução 1D com o filtro
        convolution_result = np.convolve(binary_sequence, self.filter, mode='valid')

        # Aplicar uma penalidade conforme aumenta o index dos elementos
        penalty = np.arange(len(convolution_result)) * 0.01
        convolution_result = abs(convolution_result) - penalty
        # buscando o argmax
        point_of_change = np.argmax(convolution_result) + (len(self.filter)//2)
        return point_of_change

    def cut_audio(self, audio: str, path_audios: str, time: int, path_save: str):
        """
        Essa função recorta o áudio no ponto de troca.
        :param audio: o nome do áudio
        :param path_audios: o caminho dos áudios
        :param time: o tempo que a URA acaba
        :param path_save: o caminho para salvar o áudio recortado
        :return: None
        """
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        wave, sr = librosa.load(os.path.join(path_audios, audio), sr=None, mono=False)
        wave = wave[:, time * sr:]
        wave = wave.T  # https://stackoverflow.com/a/75850841
        # salvando em mp3
        path = os.path.join(path_save, audio.replace(".opus", ".mp3"))
        sf.write(path, wave, int(sr), format='mp3')


if __name__ == '__main__':
    path_csvs_ = '../Cientificos/ExtracaoFeatures/features'
    csvs_ = ["features_1e54b932-d382-474d-8e9f-a69e733c7c73.opus.csv"]
    remover = RemoveURA(path_csvs_, csvs_)
    df = remover.run_default(5)
    # percorrendo o df e recortando os áudios
    path_audios_ = "../audios_analise_qualidade"
    path_save_cut_audios_ = r"C:\Users\alexv\PycharmProjects\AnaliseAutomatica-RemoveURA\audios_analise_qualidade_cut"
    for idx, row in df.iterrows():
        audio = row["audio"].replace("features_", "").replace(".csv", "")
        inicio_atendimento = row["inicio_atendimento"]
        # recortando o audio
        remover.cut_audio(audio, path_audios=path_audios_, time=inicio_atendimento, path_save=path_save_cut_audios_)
