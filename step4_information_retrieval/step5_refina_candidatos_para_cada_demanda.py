import pandas as pd
import os
from LLMs.ChatGPT import SimpleCompletion
from tqdm import tqdm

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"

# instanciando o objeto do chatgpt

gpt_obj = SimpleCompletion(api_key_path=f"{cwd}/key.txt")


def main(path_demandas=None):
    assert path_demandas is not None, "Passe um caminho da pasta das demandas"
    path = f"{cwd}/{path_demandas}/demandas_e_respostas_candidatas.csv"

    # Carrega o arquivo csv com as demandas e as respostas candidatas
    df = pd.read_csv(path)
    if "respostas_refinadas" in df.columns:
        respostas_refinadas = df[df['respostas_refinadas'] != "NOT YET"]['respostas_refinadas'].tolist()
    else:
        respostas_refinadas = []

    # iterando por cada linha do dataframe e extraindo a demanda e os candidatos
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # verificando se o dataframe contém a coluna respostas_refinadas
        if "respostas_refinadas" in df.columns:
            # verificando qual o primeiro not yet dessa coluna
            if df["respostas_refinadas"][index] != "NOT YET":
                continue
        demanda = row["fala_reescrita_cliente"]
        if int(index) % 10 == 0 and int(index) != 0:
            print("salvando")
            df["respostas_refinadas"] = respostas_refinadas + (["NOT YET"] * (df.shape[0] - int(index)))
            # salvando o dataframe
            df.to_csv(path, index=False)

        ################# NÃO DEU CERTO -> POUCOS DETECTADOS COMO INVÁLIDOS.
        # # step 1: verificar se a demanda é válida ou não.
        # # carregando o prompt / contexto
        # contexto = open(f"{cwd}/step4_information_retrieval/prompts/refina_prompt1.txt", "r").read()
        # # setando o contexto
        # gpt_obj.set_context(contexto)
        # # obtendo a resposta do modelo de linguagem
        # resposta1 = gpt_obj.get_completion(text=demanda)[0]
        # # realizando o append da resposta na lista validade_demandas
        # validade_demandas.append(resposta1)
        # gpt_obj.reset_messages(maintain_context=False)  # resetando as mensagens para o próximo loop
        ################################

        demanda = "<dem>" + demanda + "</dem>"
        # step 2: verificar se os candidatos são válidos ou não, caso a demanda seja válida.
        # realizando o split dos candidatos
        candidatos = row['respostas_candidatas'].split("<sep>")[:-1]

        # iterando por cada candidato e verificando se contém informações pertinentes à demanda.
        respostas_gpt = []
        # iterando apenas pelos 3 primeiros candidatos por causa de tempo
        for candidato in candidatos[:3]:
            candidato = "<res>" + candidato + "</res>"
            # carregando o prompt / contexto
            contexto = open(f"{cwd}/step4_information_retrieval/prompts/refina_prompt2_v2.txt", "r").read()
            # setando o contexto
            gpt_obj.set_context(contexto)
            # obtendo a resposta do modelo de linguagem
            resposta2 = gpt_obj.get_completion(text=demanda + "\n" + candidato)[0]
            # armazenando as respostas
            respostas_gpt.append(resposta2)
            gpt_obj.reset_messages(maintain_context=False)  # resetando as mensagens para o próximo loop

        respostas_refinadas.append("<sep>".join(respostas_gpt))

    # criando as colunas no dataframe para armazenar as validades das demandas e as respostas refinadas
    df["respostas_refinadas"] = respostas_refinadas

    # salvando o dataframe
    df.to_csv(path, index=False)
