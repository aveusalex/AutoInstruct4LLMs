# Código extraído do repositório Data-Juicer, disponível em:
# https://github.com/alibaba/data-juicer/blob/main/data_juicer/ops/mapper/punctuation_normalization_mapper.py

punctuation_unicode = {
    '，': ',',
    '。': '.',
    '、': ',',
    '„': '"',
    '”': '"',
    '“': '"',
    '«': '"',
    '»': '"',
    '１': '"',
    '」': '"',
    '「': '"',
    '《': '"',
    '》': '"',
    '´': "'",
    '∶': ':',
    '：': ':',
    '？': '?',
    '！': '!',
    '（': '(',
    '）': ')',
    '；': ';',
    '–': '-',
    '—': ' - ',
    '．': '. ',
    '～': '~',
    '’': "'",
    '…': '...',
    '━': '-',
    '〈': '<',
    '〉': '>',
    '【': '[',
    '】': ']',
    '％': '%',
    '►': '-',
}


def normalize_punctuation(text):
    text = ''.join([
        punctuation_unicode.get(c, c) for c in text
    ])
    return text
