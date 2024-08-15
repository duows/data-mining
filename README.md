# Mineração de Dados: Rei e Torre vs Rei

Este projeto tem como objetivo analisar a quantidade de jogadas necessárias para que o Rei Preto perca ou empate em uma partida de xadrez, utilizando uma base de dados específica para o cenário de Rei e Torre Branco contra Rei Preto.

## Autores

- **Henrique José de Souza**
- **Igor Augusto de Carvalho**

## Professor Orientador

- **Murilo Varges da Silva**

## Base de Dados

- **Descrição**: A base de dados utilizada neste projeto contém posições finais de partidas de xadrez onde o Rei Branco e a Torre Branca enfrentam o Rei Preto. As posições são descritas pelas coordenadas das peças no tabuleiro.
  
- **Origem**: A base de dados foi descrita por Clarke em 1977. As condições de vitória foram documentadas por Michael Bain e Artur Hoff entre 1992-1994.

- **Objetivo**: Analisar a quantidade de jogadas que devem ser feitas para o Preto perder ou empatar.

### Estrutura da Base de Dados

A base de dados possui as seguintes colunas:

- Coluna do Rei Branco (Independente)
- Linha do Rei Branco (Independente)
- Coluna da Torre Branca (Independente)
- Linha da Torre Branca (Independente)
- Coluna do Rei Preto (Independente)
- Linha do Rei Preto (Independente)
- Número de passos ideais para a vitória do branco ou indicação de empate (Dependente)

## Pré-processamento

### Estratégias Adotadas

- Simulação e exclusão de dados faltantes
- Redução da base de dados para melhor visualização usando PCA (Análise de Componentes Principais)
- Transformação de Dados:
  - Padronização
  - Normalização (z-score)

## Análises Realizadas

- **PCA (2D e 3D)**: Análise para converter objetos com atributos possivelmente correlacionados em um conjunto de atributos linearmente descorrelacionados.
- **Clustering**:
  - **GMM (Gaussian Mixture Model)**: Modelo de agrupamento utilizado para identificar padrões.
  - **K-Means**: Técnica de agrupamento não supervisionada para identificar grupos na base de dados.
- **Classificação**:
  - **Árvore de Decisão**
  - **Random Forest**
  - **KNN (K-Nearest Neighbors)**
  - **SVM (Support Vector Machine)**
  - **Redes Neurais**: Comparação entre diferentes configurações de redes neurais.

## Resultados

### Comparação entre Métodos

| Método            | Acurácia | Precisão | F1    | Tempo de Execução (s) |
|-------------------|----------|----------|-------|-----------------------|
| Decision Tree     | 0.7211   | 0.72     | 0.72  | 83.16                 |
| Random Forest     | 0.6868   | 0.68     | 0.68  | 14.56                 |
| SVM               | 0.5357   | 0.58     | 0.57  | 12.86                 |
| SVM Cross Val.    | 0.6502   | 0.67     | 0.66  | 3795.33               |
| KNN               | 0.5344   | 0.48     | 0.49  | 234.95                |
| KNN Grid Search   | 0.5459   | 0.48     | 0.49  | 996.48                |

### Redes Neurais

| Configuração              | Acurácia | Precisão | F1    | Tempo de Execução (s) |
|---------------------------|----------|----------|-------|-----------------------|
| 1 Camada, 50 Neurônios     | 0.5856   | 0.5386   | 0.5424| 204.84                |
| 5 Camadas, 50 Neurônios    | 0.7215   | 0.6524   | 0.6486| 226.11                |
| 21 Camadas, 125 Neurônios  | 0.5481   | 0.4330   | 0.3910| 536.18                |
| 21 Camadas, 125 Neurônios (com dropout) | 0.6275 | N/A | N/A | N/A |

## Conclusão

Este projeto explora diferentes técnicas de mineração de dados, clustering e classificação em um cenário específico de xadrez. As redes neurais e métodos de classificação apresentaram variações significativas em termos de acurácia, precisão e tempo de execução, com a Árvore de Decisão e as redes neurais de médio porte mostrando os melhores desempenhos.

## Agradecimentos

Agradecemos ao professor Murilo Varges da Silva pela orientação durante o desenvolvimento deste projeto.
