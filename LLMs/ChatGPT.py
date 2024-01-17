import os
import openai
import logging
import datetime
from time import sleep

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


class SimpleCompletion:
    def __init__(self, api_key_path: str, logs_path: str = f"{cwd}/logs/gpt"):
        self.openai_client = openai.OpenAI(api_key=open(api_key_path, "r").read())
        self.tokens_in = 0
        self.tokens_out = 0
        self.messages = []

        # criando a pasta de logs
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # criando o logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        datet = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(f"{logs_path}/GPT_{datet}.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_context(self, context: str) -> None:
        """
        Função para definir o contexto da conversa.
        :param context: contexto da conversa str ou path_audio apontando para um arquivo de texto
        :return: None
        """
        # verificando se o contexto aponta para um arquivo de texto
        if os.path.exists(context):
            with open(context, "r") as f:
                context = f.read()

        self.messages.append({'role': 'system', 'content': context})

    def reset_messages(self, maintain_context=True):
        if maintain_context:
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def get_completion(self, text: str = None, model="gpt-3.5-turbo", temperature=0, system=None) -> tuple:
        """
        Função para obter a resposta do modelo de linguagem.
        :param text: o texto que será enviado para o modelo de linguagem (opcional)
        :param model: o modelo de linguagem que será utilizado (4k prompt padrão)
        :param temperature: nível de aleatoriedade da resposta do modelo de linguagem
        :param system: não mexer. Parâmetro interno para trocar o modelo caso o modelo atual esteja com problemas
        :return: tuple(str, openai.Response)
        """
        if text is not None:
            self.messages.append({'role': 'user', 'content': text})

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=self.messages,
                temperature=temperature,  # this is the degree of randomness of the model's output
            )
        except openai.BadRequestError as e:
            if system is None:
                self.logger.info(f"Erro ao enviar a mensagem. Trocando de modelo. Error: {e}")
                # trocando o modelo por um maior
                return self.get_completion(text, model="gpt-3.5-turbo-1106", temperature=temperature, system=1)
            else:
                self.logger.info(f"Mensagem extremamente grande. Error: {e}")
                return "BIG ERROR", None
        except openai.RateLimitError as e:
            self.logger.info(f"Erro ao enviar a mensagem. Esperando 45 segundos. Error: {e}")
            sleep(45)
            return self.get_completion(text, model=model, temperature=temperature)

        modelo = response.model
        idd = response.id
        texto_resp = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        self.logger.info(f"Modelo: {modelo} | ID: {idd} | Texto: {texto_resp} | Tokens_in: {tokens_in} | Tokens_out: {tokens_out}")
        return response.choices[0].message.content, response


class Chat(SimpleCompletion):
    def __init__(self, api_key_path: str):
        super().__init__(api_key_path)
        self.messages = []

    def add_context(self, context: str):
        # inserindo o contexto no começo da lista de mensagens
        context = {'role': 'system', 'content': context}
        self.messages = [context] + self.messages

    def add_message(self, role: str, prompt: str):
        self.messages.append({'role': role, 'content': prompt})
