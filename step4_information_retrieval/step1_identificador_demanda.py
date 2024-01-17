# esse código visa identificar as demandas de forma automática, utilizando o ChatGPT, a partir das transcrições do
# cliente e reescrevê-las de forma mais clara e objetiva.

from LLMs import ChatGPT
import os
import pandas as pd
import logging
import datetime
from step2_text_preprocessing import main_preprocessing as preprocessing
from tqdm import tqdm

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


class IdentificadorDemanda:
    """
    Primeiro: carregar as transcrições em CSV
    Segundo: Selecionar apenas as falas do cliente
    Terceiro: Pedir ao chatgpt identificar possíveis demandas nas falas do cliente
    Quarto: Salvar as demandas identificadas em um CSV
    """
    def __init__(self, transcricoes_path: str = None, api_key_path: str = "./key.txt",
                 logs_path: str = f"{cwd}/logs/id_demanda"):
        self.transcricoes = [trans for trans in os.listdir(transcricoes_path) if trans.endswith(".csv")]
        self.path_transcricoes = transcricoes_path
        self.api_key_path = api_key_path
        self.logs_path = logs_path
        # instanciando objetos
        self.chatgpt = ChatGPT.SimpleCompletion(api_key_path)
        self.prepro_engine = preprocessing.Preprocessamento()
        self.logger = self.__criar_logger()
        self.demandas = self.__load_demandas()
        self.ja_foram = [dem[0] for dem in self.demandas]
        self.tokens_in = 0
        self.tokens_out = 0

    def identificar_demandas(self, path_to_save=None):
        """
        Essa função carrega as transcrições, executa o pré-processamento do texto e pede ao ChatGPT para identificar
        as demandas.
        :return:
        """
        assert path_to_save is not None, "Você deve especificar qual o path para salvar as demandas"
        for transc in tqdm(self.transcricoes):
            # verificando se a transcrição já foi processada
            if transc in self.ja_foram:
                self.logger.info(f" Arquivo {transc} já foi processado!")
                continue
            # carregando as transcrições
            df = pd.read_csv(f"{self.path_transcricoes}/{transc}")  # seria interessante usar parquet
            falas_cliente = self.__falas_cliente(df)
            # realizando o pré-processamento do texto
            falas_cliente = self.prepro_engine.preprocessar_texto(falas_cliente, cliente_id=transc)
            falas_cliente = "\n\n<transcription>\n" + falas_cliente + "\n</transcription>"
            # enviando as falas do cliente para o ChatGPT
            context = open(f"{cwd}/step4_information_retrieval/prompts/promptV2.txt", "r").read()  # definindo o contexto / Prompt
            context = context + falas_cliente
            fala_reescrita, etc = self.chatgpt.get_completion(context)
            # resetando as mensagens
            self.chatgpt.reset_messages()
            if etc is not None:
                self.tokens_in += etc.usage.prompt_tokens
                self.tokens_out += etc.usage.completion_tokens
            # salvando as demandas reescritas
            self.demandas.append((transc, falas_cliente, fala_reescrita))
            self.ja_foram.append(transc)
            # logs
            self.logger.info(f" Arquivo: {transc} - Fala reescrita: {fala_reescrita}")
            if len(self.demandas) % 10 == 0:
                self.logger.info(f" Salvando demandas...")
                self.__salvar_demandas(path_to_save)
        # salvando as demandas
        self.__salvar_demandas(path_to_save)

    def __load_demandas(self, path: str = f"{cwd}/dados/gold/demandas_identificadas"):
        """
        Função para carregar as demandas identificadas
        :return: None
        """
        try:
            df = pd.read_csv(f"{path}/demandas_identificadas.csv")
            self.logger.info(f" Demandas carregadas!")
            return [(row[0], row[1], row[2]) for row in df.values]
        except FileNotFoundError:
            self.logger.info(f" Nenhuma demanda encontrada!")
            return []

    def __salvar_demandas(self, path: str = f"{cwd}/dados/gold/demandas_identificadas"):
        """
        Função para salvar as demandas identificadas
        :param path: path_audio para salvar as demandas
        :return: None
        """
        # criando a pasta de logs
        if not os.path.exists(f"{cwd}/{path}"):
            os.makedirs(f"{cwd}/{path}")
        # gerando csv
        df = pd.DataFrame(self.demandas, columns=['arquivo', 'fala_cliente', 'fala_reescrita'])
        df.to_csv(f"{cwd}/{path}/demandas_identificadas.csv", index=False)
        # salvando os logs
        self.logger.info(f" Finalizado!")
        self.logger.info(f" TOTAL Tokens in: {self.tokens_in} | Preço: {(self.tokens_in/1000) * 0.0015}")
        self.logger.info(f" TOTAL Tokens out: {self.tokens_out} | Preço: {(self.tokens_in/1000) * 0.002}")

    def __falas_cliente(self, transcricao: pd.DataFrame) -> str:
        """
        Função para selecionar apenas as falas do cliente
        :param transcricao: dataframe com as transcrições
        :return: str com as falas do cliente
        """
        return transcricao[transcricao['role'] == 'cliente']['transcription'].str.cat(sep=" ")

    def __criar_logger(self):
        """
        Função para criar o logger
        """
        # criando a pasta de logs
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        # criando o logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        datet = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(f"{self.logs_path}/IdentificadorDemanda_{datet}.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


if __name__ == '__main__':
    idd = IdentificadorDemanda("../dados/gold/deprecated/anatel_transcricoes", api_key_path="../key.txt")
    idd.identificar_demandas()
