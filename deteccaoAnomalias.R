#
# Detecção de Anomalias.
# 
# 

# Carga das Bibliotecas ----
library(keras)
library(dplyr)
library(ggplot2)

setwd("~/Sistemas/GitHub/anomalias")

options(scipen = 999)

# Carga dos dados  ----
df <- read.csv("creditcard.csv", header = T) 

# Equalizando o conjunto de dados para rebalancear as classes de Verdadeiras/Fraudes

# Separando compras verdadeiras de compras frudulentas
df_real   <- df %>% filter(Class==0)
df_fraude <- df %>% filter(Class==1)

# Selecionando observacoes para o dataset de observacoes verdadeiras
set.seed(1)
tam_real  <- nrow(df_fraude) * 1
df_real   <- sample_n(df_real, tam_real)

# Gerando novo conjunto de dados balanceado
all_data <- rbind(df_real,df_fraude)

# Normalizando o atrubto valor
all_data$Amount <- scale(all_data$Amount)

# Retirando as variáveis indesejadas
clean_data  <- all_data  %>% 
               select(-c(Time))   %>%
               as.data.frame()

# Converte todas as colunas para tipo numerico
clean_data <- sapply(clean_data, as.numeric) %>% as.matrix()

# Divisao dos dados em conjunto de treino e teste ----
tamanho_treino <- floor(0.8 * nrow(clean_data))
treino_ind     <- sample(seq_len(nrow(clean_data)), size = tamanho_treino)

treino <- clean_data[treino_ind, ] %>% as.matrix()
teste  <- clean_data[-treino_ind,] %>% as.matrix()

# Definicao dos parametros do modelo autoencoder
input   <- output <- ncol(clean_data)

dropOut <-  0.2
atvn    <-  "elu"
batch   <-  32
epochs  <-  250

# Modelo de Auto encoder  ----
# O modelo de auto encoder usado é um modelo simétrico.
# Camada Input: Inclui o formato de entrada do dataset de treino, ou seja, 30 features
# Camadas Encoder: Codificamos a camada de input com 4 camadas e normalização por batch e dropout.
# Camadas Decoder: Simetrica a camada de codificação.

C1 <- 1024   # 256
C2 <- 128    # 128
C3 <- 64     #  64
C4 <- 32     #  32

# Cria a arquitetura do  autoencoder
input_layer <- layer_input(shape = c(input))

encoder <-  input_layer                                %>% 
            layer_dense(units = C1, activation = atvn) %>%
            layer_batch_normalization()                %>%
            layer_dropout(rate = dropOut)              %>%
            layer_dense(units = C2, activation = atvn) %>%
            layer_dropout(rate = dropOut)              %>%
            layer_dense(units = C3, activation = atvn) %>%
            layer_dense(units = C4) 

decoder <-  encoder                                    %>% 
            layer_dense(units = C3, activation = atvn) %>% 
            layer_dropout(rate = dropOut)              %>% 
            layer_dense(units = C2, activation = atvn) %>%
            layer_dropout(rate = dropOut)              %>%
            layer_dense(units = C1, activation = atvn) %>%
            layer_dense(units = output) #

# Combina as camadas de encoder e decoder
autoencoder_model <- keras_model(inputs  = input_layer, 
                                 outputs = decoder)

# Compila o modelo
autoencoder_model %>% 
  compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')

# Examina o modelo
summary(autoencoder_model)

# Processa o treinamento do modelo
history <- autoencoder_model %>%
            keras::fit(treino,
                       treino,
                       epochs  = epochs,
                       shuffle = TRUE,
                       batch = batch,
                       validation_data = list(teste, teste))

# Imprime evolução do modelo converg
plot(history)

# Função que calcula o erro de reconstrução
reconstMSE = function(i){
  reconstructed_points = autoencoder_model %>% 
    predict(x = data[i,] %>% 
              matrix(nrow = 1, ncol = output)
    )
  return(mean((data[i,] - reconstructed_points)^2))
}

# Inicialmente verifica o dataset treino
data  = treino
lines = nrow(treino)

# Calcula o erro de reconstrução dos dadosde treino
treinoRecon = data.frame(data = treino, 
                        score = do.call(rbind, lapply(1:lines, FUN = reconstMSE)))

# Calcula o limite para considerar a observação anomalia em 98%
anomalyLimit = quantile(treinoRecon$score, p = 0.98)

# Verifica o dataset de teste
data = teste
lines = nrow(data)

# Calcula o erro de reconstrução dos dadosde teste
testeRecon = data.frame(data = teste, 
                       score = do.call(rbind, lapply(1:lines, FUN = reconstMSE)))

# Combina os erros de treino e de teste 
Recondata         <- rbind(treinoRecon, testeRecon)
Recondata$anomaly <- ifelse(Recondata$score > anomalyLimit, 'Anomalia', 'Normal')
Recondata$Classe  <- ifelse(Recondata$data.Class==0, 'Normal', 'Fraude')

# Cria matrix de cruzamento de resultados
table(Recondata$anomaly, Recondata$Classe)

# Cria datasets para com as classes de Fraude e Normal
RecondataFraude <- Recondata %>% filter(Classe=='Fraude')
RecondataNormal <- Recondata %>% filter(Classe=='Normal')

# Imprime os resultados

# Anomalias detectadas na classe fraude
plot(RecondataFraude$score, 
     col = ifelse(RecondataFraude$score > anomalyLimit, "red", "green"), 
     pch = 19,
     main = "Anomalias Detectadas (Classe Fraude)",
     xlab = "observações", 
     ylab = "score")

abline(h = anomalyLimit, 
       col = "red", 
       lwd = 1)

# Anomalias detectadas na classe normal
plot(RecondataNormal$score, 
     col = ifelse(RecondataNormal$score > anomalyLimit, "red", "green"), 
     pch = 19, 
     main = "Anomalias Detectadas (Classe Normal)",
     xlab = "observações", 
     ylab = "score")

abline(h = anomalyLimit, 
       col = "red", 
       lwd = 1)

# Anomalias detectadas no conjunto completo
plot(Recondata$score, 
     col = ifelse(Recondata$score > anomalyLimit, "red", "green"), 
     pch = 19, 
     main = "Anomalias Detectadas",
     xlab = "observações", 
     ylab = "score")

abline(h = anomalyLimit, 
       col = "red", 
       lwd = 1)


