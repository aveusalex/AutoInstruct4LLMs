# esse código é responsável por realizar o pré-processamento de textos passados para ele, baseando-se nos problemas
# pertinentes ao projeto.
from step2_text_preprocessing import trans_repetition_error
from step2_text_preprocessing import fix_unicode_mapper
from step2_text_preprocessing import punctuation_normalization_mapper
from step2_text_preprocessing import remove_nomes
import os
import logging
import datetime
import re


cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


class Preprocessamento:
    """
    Classe responsável por realizar o pré-processamento de textos.
    Suas etapas são:
    1) Remoção de repetições equidistantes de palavras e caracteres.
    2) Correção de caracteres unicode.
    3) Normalização de pontuação.
    4) Remoção de números.
    5) Remoção de nomes.
    """
    def __init__(self, logs_path: str = f"{cwd}/logs/preprocessamento_texto"):
        # criando a pasta de logs
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # instanciando objeto de remoção de nomes
        self.remove_nomes = remove_nomes.NomeIdentifier()

        # criando o logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        datet = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(f"{logs_path}/preprocessamento_{datet}.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def preprocessar_texto(self, texto: str, cliente_id: str) -> str:
        """
        Função para realizar o pré-processamento de um texto.
        :param texto: texto que será pré-processado
        :param cliente_id: id do cliente para logging
        :return: texto pré-processado
        """
        original_text = texto
        text = trans_repetition_error.filtra_repeticoes_equidistantes(texto)
        text = fix_unicode_mapper.fix_unicode(text)
        text = punctuation_normalization_mapper.normalize_punctuation(text)
        # anonimizando CPFs
        text = self.remove_numeros(text)
        # anonimizando nomes
        text = self.remove_nomes.remover(text)
        self.logger.info(f"\nID: {cliente_id}\nTexto original: <<{original_text}>>\nTexto pré-processado: <<{text}>>\n\n")
        return text

    def remove_numeros(self, texto: str) -> str:
        """
        Função para remover os numeros presente no texto.
        :param texto: texto que será pré-processado
        :return: texto pré-processado
        """
        # removendo usando re
        texto = re.sub(r'\d+', '', texto)
        return texto
