import openai
import logging
import os
import datetime

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


class Ada002:
    def __init__(self, api_key_path: str, logs_path: str = f"{cwd}/logs/embedding"):
        self.openai_client = openai.OpenAI(api_key=open(api_key_path, "r").read())
        self.tokens_in = 0

        # verificando se a pasta de logs existe
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # criando o logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        datet = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(f"{logs_path}/embedding_{datet}.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_embedding(self, text: str) -> tuple:
        """
        Função para obter o embedding de um texto.
        Retorna o embedding e o número de tokens usados até o momento.
        :param text: texto que será enviado para o modelo
        :return: tuple(list, int)
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )

        response_emb = response.data[0].embedding
        response_usage = response.usage.total_tokens
        preco = response_usage * 0.0001/1000
        # registrando o log
        self.logger.info(f" Tokens usados: {response_usage} - Preço: U${preco:.3f} - Texto: {text}")
        self.tokens_in += response_usage

        return response_emb, self.tokens_in


if __name__ == '__main__':
    obj = Ada002(api_key_path="../key.txt")
    print(len(obj.get_embedding("Hello, my name is Alex.")[0]))
    print(len(obj.get_embedding("Hello, my name is Mariana.")[0]))
