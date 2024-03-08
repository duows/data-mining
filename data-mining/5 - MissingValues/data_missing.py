import random

def alterar_porcentagem(base_dados, porcentagem_para_alterar):
    # Convertendo a string da base de dados para uma lista de tuplas
    dados = [tuple(item.split(',')) for item in base_dados.split(',')]

    # Calculando o número de itens a serem alterados
    total_itens = len(dados)
    itens_para_alterar = int(total_itens * (porcentagem_para_alterar / 100))

    # Alterando aleatoriamente valores para "?"
    indices_para_alterar = random.sample(range(total_itens), itens_para_alterar)
    for indice in indices_para_alterar:
        dados[indice] = ('?', '?', '?')  # Modifique de acordo com a estrutura dos seus dados

    # Convertendo a lista de tuplas de volta para a string
    nova_base_str = ','.join([','.join(map(str, item)) for item in dados])

    return nova_base_str

# Leitura do arquivo
with open('krkopt.data', 'r') as arquivo:
    base_dados_original = arquivo.read()

# Porcentagem de dados para alterar
porcentagem_para_alterar = 3

# Alterando a porcentagem de dados e obtendo a nova base de dados
nova_base_dados = alterar_porcentagem(base_dados_original, porcentagem_para_alterar)

# Salvando a nova base de dados em um novo arquivo .data
with open('krkopt_altered.data', 'w') as novo_arquivo:
    novo_arquivo.write(nova_base_dados)

print('rodando..')
