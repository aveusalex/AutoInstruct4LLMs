import pandas as pd
import os
from LLMs.ChatGPT import SimpleCompletion
from tqdm import tqdm
import threading

cwd = os.getcwd().split("AutoInstruct4LLMs")[0] + "AutoInstruct4LLMs/"


def main(n_threads: int = 1, path_demandas=None):
    assert path_demandas is not None, "Defina o caminho para a pasta de demandas."
    # instanciando o objeto do chatgpt
    gpts = [SimpleCompletion(api_key_path=f"{cwd}/key.txt") for _ in range(n_threads)]

    path = f"{cwd}/{path_demandas}/demandas_e_respostas_candidatas.csv"

    # Carrega o arquivo csv com as demandas e as respostas candidatas
    df = pd.read_csv(path)

    # filtrando apenas os resultados
    df = df.drop(columns=["respostas_candidatas", "tempo_busca"])

    # carregando o prompt / contexto
    contexto = open(f"{cwd}/step4_information_retrieval/prompts/combina_prompt_v2.txt", "r").read()
    # setando o contexto
    [gpt.set_context(contexto) for gpt in gpts]

    # definindo a lista que armazena as respostas unificadas
    respostas_unificadas = []

    # iterando por cada linha do dataframe e extraindo as respostas refinadas
    threads = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if int(index) % n_threads == 0 and int(index) != 0:  # salvando a cada 50 iterações
            [t.join() for t in threads]
            save(df, respostas_unificadas, path_demandas)
            threads = []

        t = threading.Thread(target=combina_respostas_refinadas, args=(index, row, respostas_unificadas, gpts[index % n_threads]))
        t.start()
        threads.append(t)

    [t.join() for t in threads]
    save(df, respostas_unificadas, path_demandas)

    gpts_use = [[gpt.tokens_in, gpt.tokens_out] for gpt in gpts]
    print(f"Tokens usados: {sum([gpt[0] for gpt in gpts_use])} - Preço: U${sum([gpt[0] for gpt in gpts_use]) * 0.0015/1000:.3f}")
    print(f"Tokens gerados: {sum([gpt[1] for gpt in gpts_use])} - Preço: U${sum([gpt[1] for gpt in gpts_use]) * 0.002/1000:.3f}")


def combina_respostas_refinadas(indx, row, lista_respostas_refinadas, gpt_obj):
    respostas_refinadas = row["respostas_refinadas"].split("<sep>")  # separando as respostas refinadas (str -> list)
    # step 1: pedir ao gpt combinar as respostas refinadas
    # enumerando cada resposta e transformando tudo em uma string
    respostas_refinadas = "\n".join([f"{i + 1}) {resposta}" for i, resposta in enumerate(respostas_refinadas)])

    # obtendo a resposta do modelo de linguagem
    resposta1 = gpt_obj.get_completion(text=respostas_refinadas)[0]

    # realizando o append da resposta na lista de respostas refinadas
    lista_respostas_refinadas.append((indx, resposta1))
    gpt_obj.reset_messages(maintain_context=True)  # resetando as mensagens para o próximo loop


def save(df, respostas_unificadas, path):
    # ordenando as respostas unificadas pelo índice
    respostas_unificadas = sorted(respostas_unificadas, key=lambda x: x[0])
    # removendo o índice
    respostas_unificadas = [resposta[1] for resposta in respostas_unificadas]
    # adicionando as respostas unificadas ao dataframe
    df["respostas_unificadas"] = respostas_unificadas + (["NOT YET"] * (df.shape[0] - len(respostas_unificadas)))
    # salvando o dataframe
    df.to_csv(f"{cwd}/{path}/demandas_respostas_final.csv", index=False)


if __name__ == '__main__':
    main(n_threads=20)
