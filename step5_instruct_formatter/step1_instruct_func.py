from random import sample
import os

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


def sample_instruction() -> str:
    """
    Essa função seleciona aleatoriamente uma instrução e a retorna.
    :return: instrução selecionada aleatoriamente
    """
    instructions = ["Com seus conhecimentos, responda à seguinte solicitação de um cliente:",
                    "Elabore uma resposta para a demanda do usuário:",
                    "Apresente uma resposta objetiva para a seguinte demanda:",
                    "Responda à solicitação do cliente levando em conta seus conhecimentos:",
                    "Responda de forma objetiva e direta a demanda de um cliente de call center:",
                    "Levando em consideração o contexto de call center e telecomunicações, responda a demanda:",
                    "Leia atentamente à pergunta de um cliente e a responda em seguida:",
                    "Utilizando seus conhecimentos sobre call center, responda à pergunta a seguir:",
                    "Assuma o papel de um atendente de call center e responda à demanda de um cliente:",
                    "Você é um atendente de call center e um cliente te fez uma pergunta. Responda-a:"]
    instruction = sample(instructions, 1)[0]
    return instruction


def add_instruction_to_dataframe(df_path: str) -> None:
    import pandas
    """
    Essa função adiciona uma coluna de instruções a um dataframe.
    :param df_path: caminho do dataframe
    """
    df_path = cwd + df_path
    df = pandas.read_csv(df_path)
    df['instruction'] = ""
    df["instruction"] = df["instruction"].apply(lambda x: sample_instruction())
    df = df[["arquivo", "instruction", "fala_reescrita_cliente", "respostas_unificadas"]]
    #adicionando um marcador no nome do arquivo para indicar que ele já foi processado
    df_path = df_path.split(".csv")[0] + "_instruct.csv"
    df.to_csv(df_path, index=False)
