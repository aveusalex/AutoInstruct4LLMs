# esse código é responsável por analisar os textos e identificar os nomes presentes neles, substituindo por NOME
import re
import os
import pandas as pd
cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"

"""
Disclaimer:

Tentei utilizar a biblioteca do SPACY como forma de identificar os nomes, mas não estava funcionando bem, identificando
algumas palavras equivocadamente como nomes e não identificando alguns nomes como tal. Por isso, optei por utilizar
uma lista de nomes brasileiros extraída do IBGE, que contém nomes reais.

No entanto, essa lista estava com alguns nomes muito diferentes e específicos, como "claro", "nao", que faziam com que
outras palavras fossem identificadas como nomes, equivocadamente. 

Por isso, optei por utilizar apenas os 500 nomes mais comuns, que são mais genéricos e não causam esse problema.

MELHORIA:
Uma forma de melhorar a abrodagem com spacy é utilizando o NLTK para avaliar a estrutura sintática da frase antes de
usar o spacy para determinar a função das palavras.

Um tutorial a ser seguido é https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
"""

class NomeIdentifier:
    def __init__(self):
        self.lista_nomes = self.__carrega_nomes()

    def remover(self, texto):
        """
        Função para remover os nomes presentes no texto.
        :param texto: texto que será pré-processado
        :return: texto pré-processado
        """
        for word in texto.split():
            # removendo pontuação
            word = re.sub(r'[^\w\s]', '', word)
            original_word = word
            nome_acentuado = word

            # substituindo acentos e caracteres especiais por suas formas sem acento
            word = re.sub(r'[áàãâä]', 'a', word)
            word = re.sub(r'[éèêë]', 'e', word)
            word = re.sub(r'[íìîï]', 'i', word)
            word = re.sub(r'[óòõôö]', 'o', word)
            word = re.sub(r'[úùûü]', 'u', word)
            word = re.sub(r'[ç]', 'c', word)
            nome = word.lower()
            nome_acentuado = nome_acentuado.lower()
            if nome in self.lista_nomes or nome_acentuado in self.lista_nomes:
                texto = texto.replace(original_word, "NOME")
        return texto

    @staticmethod
    def __carrega_nomes():
        """
        Função para carregar a lista de nomes.
        :return: lista de nomes
        """
        # usando apenas os 500 nomes mais comuns
        df_mas = pd.read_csv(f"{cwd}/dados/gold/nomes-brasileiros-ibge/ibge-mas-10000.csv", nrows=500)
        df_fem = pd.read_csv(f"{cwd}/dados/gold/nomes-brasileiros-ibge/ibge-fem-10000.csv", nrows=500)
        lista_nomes = list(df_mas['nome']) + list(df_fem['nome'])
        lista_nomes = [nome.lower() for nome in lista_nomes]

        return lista_nomes
