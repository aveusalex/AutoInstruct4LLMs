import time


def filtra_repeticoes_equidistantes(frase: str) -> str:
    """
    Encontra as repetições de palavras que são equidistantes em uma frase. Para esse filtro funcionar, ele parte
    do pressuposto que as repetições de palavras que são equidistantes são alucinações do Whisper.
    :param frase: frase a ser analisada.
    :return: frase sem as repetições equidistantes.
    """
    palavras = frase.split()
    repeticoes = {}

    # criando um dicionário com as palavras e seus respectivos índices
    for i, palavra in enumerate(palavras):
        if palavra not in repeticoes:
            repeticoes[palavra] = [i]
        else:
            repeticoes[palavra].append(i)

    # instanciando a lista que receberá os índices das palavras que se repetem equidistantemente
    resultado = []

    # iterando sobre o dicionário de repetições, selecionando apenas as palavras que se repetem pelo menos 4 vezes
    for palavra, indices in repeticoes.items():
        distancias_repeticoes = {}
        if len(indices) >= 4:  # selecionando apenas as palavras que se repetem pelo menos 4 vezes
            # verificando a distância entre as repetições. Se for equidistante, será adicionado na lista de resultados
            for i in range(1, len(indices)):
                dist = indices[i] - indices[i - 1]  # calculando a distância entre as repetições
                if dist not in distancias_repeticoes:
                    distancias_repeticoes[dist] = [indices[i - 1], indices[i]]
                else:
                    distancias_repeticoes[dist].append(indices[i - 1])
                    distancias_repeticoes[dist].append(indices[i])

            # verificando quantas repetições equidistantes tem para cada distância
            for _, indices in distancias_repeticoes.items():
                if len(set(indices)) >= 5:  # arbitrário, limiar de repetições equidistantes para determinar se é uma alucinação
                    resultado += list(set(indices))

    # filtrando as palavras que se repetem equidistantemente
    for idx in resultado:
        palavras[idx] = ""
    palavras = [palavra for palavra in palavras if palavra != ""]

    return " ".join(palavras)


if __name__ == '__main__':
    frase = "Isso é um teste é um teste é um teste é um teste é um teste de equidistante de repetições em um teste."
    with open("../aux.txt", encoding="utf-8") as f:
        frase2 = f.read()
    start = time.time()
    res = filtra_repeticoes_equidistantes(frase)
    end = time.time()
    print(f"Tamanho frase: {len(frase)}\nTempo: {end - start:.3f}\n")
    print(res)

    start = time.time()
    res2 = filtra_repeticoes_equidistantes(frase2)
    end = time.time()
    print(f"Tamanho frase 2: {len(frase2.split())}\nTempo: {end - start:.3f}\n")
    print(res2)

