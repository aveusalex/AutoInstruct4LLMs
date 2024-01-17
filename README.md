# Auto Instruct for LLMs

## Gerador de datasets instrutivos para treino de LLMs a partir de ligações de call center.

**Por Alex Echeverria**, Tese de Conclusão de Curso do Bacharelado em Inteligência Artificial.

---

### Como usar esta ferramenta:

Siga os passos abaixo:

1. **Instalando os requirements:**
   - Requisito de python: 3.10.0
   - Recomendo o uso de conda para criar environments.
     - A partir de um shell / terminal, crie um environment conda com o seguinte comando:
       ```bash
       conda create --name AutoInstruct4LLMs python=3.10
       ```
     - Instale as bibliotecas necessárias, usando o comando:
       ```bash
       pip install -r requirements.txt
       ```
     - Instale o ffmpeg, com o seguinte comando:
       ```bash
       conda install conda-forge::ffmpeg
       ```

2. **Preparando os dados:**
   - Coloque todos os arquivos de áudio que possui no diretório "./dados/bronze/{nome_operação}" (crie um, se não houver). (nome operação é o nome da empresa de call center que deseja aplicar a ferramenta)
   - Insira sua chave da OpenAI (token) em um txt de nome "key.txt", na raiz do diretório (direto em "./AutoInstruct4LLMs")

3. **Configurando o Elasticsearch:**
   - Crie uma instância elasticsearch (recomendo usar docker), determine o nome do índice como sendo "{OPERACAO}_index".
   - Salve a sua senha do elasticsearch na variável password_elastic (linha 36 do código main.py) e salve o arquivo http_ca.crt também na raiz do repositório.  

4. **Executando o script principal:**
   - Execute o script principal em main.py, substituindo o valor de OPERACAO pelo nome da operação que deseja usar.

5. **Obtendo o dataset:**
   - Após a finalização da execução, o dataset estará disponível em "./dados/gold/demandas_{OPERACAO}/demandas_respostas_final_instruct.csv"
