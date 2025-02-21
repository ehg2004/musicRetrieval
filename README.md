# Recuperação de Músicas com Aprendizado de Máquina

## Sobre o Projeto
Este projeto foi desenvolvido como parte da disciplina **Introdução à Modelagem e Aprendizado** no curso de **Engenharia de Computação** da **UTFPR - Curitiba**. O objetivo é explorar métodos de **Aprendizado de Máquina** aplicados à recuperação de músicas, inspirando-se no funcionamento do Shazam.

## Tecnologias Utilizadas
- **Python** para web scraping e extração de features
- **Bibliotecas de Processamento de Áudio**: TSFEL e Open3L
- **Orange3** para análise de dados e modelagem
- **Selenium e Youtube-dl** para obtenção de áudios
- **Redução de Dimensionalidade** com PCA
- **Classificação e Clusterização** utilizando K-Means, Regressão Logística e outras técnicas

## Metodologia
1. **Coleta de Dados**: Uso de web scraping para baixar listas de músicas populares e extrair os áudios do YouTube.
2. **Extração de Features**: Comparação entre métodos estatísticos (TSFEL) e redes neurais (Open3L) para obtenção de embeddings de áudio.
3. **Análise de Clusters**: Aplicação de K-Means e avaliação da separabilidade dos dados.
4. **Classificação**: Treinamento de modelos para associar músicas baseando-se nas features extraídas.
5. **Redução de Dimensionalidade**: Uso de PCA e seleção de features para otimizar o desempenho dos modelos.
6. **Validação**: Comparação dos resultados obtidos pelos diferentes métodos e análise qualitativa das previsões.

## Resultados
- Extração de features resultou em um conjunto de **3500 músicas** para análise.
- **Modelos baseados em Open3L** apresentaram melhor agrupamento das músicas.
- **Redução de Dimensionalidade** melhorou o desempenho dos classificadores.
- O sistema identificou **similaridades musicais inesperadas**, demonstrando a viabilidade do método.

## Como Executar o Projeto
1. **Instalar Dependências**:
   ```sh
   pip install selenium youtube-dl tsfel orange3
   ```
2. **Executar o Web Scraping**:
   ```sh
   python scrape_songs.py
   ```
3. **Processar os Áudios**:
   ```sh
   python extract_features.py
   ```
4. **Treinar e Avaliar os Modelos**:
   ```sh
   python train_model.py
   ```

## Autores
- **Enzo Holzmann Gaio**
- **Orientador: Prof. Dr. Heitor Silvério Lopes**

## Licença
Este projeto está licenciado sob a licença **MIT**.

