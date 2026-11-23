# Rutas y flags
CC = gcc
BIN = ./bin
SRC = ./src
EJEMPLOS = ./ejemplos

# Directorios (Cambiar LLAMA_DIR donde se instala llama.cpp
LLAMA_DIR = ../llama.cpp
LLAMA_LIB_DIR = $(LLAMA_DIR)/build
 
#FLags de compilación
CCFLAGS = -O2 -Wall -fopenmp

# Rutas cabeceras
LLAMA_INC = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/common
GESTOR_INC = -I$(SRC) 
# Otros includes de otras librerias segun proyectos
OTROS_INC =-I./auxiliares -I/usr/include/curl -I/usr/include/cjson -I/usr/include/libxml2 

# Analisis de leakes
ANALISIS = -fsanitize=address -g 

# Fuentes Gestor
GESTOR_SRC = $(SRC)/*.c ./auxiliares/*.c 

# Librerias
LDFLAGS = -L$(LLAMA_LIB_DIR)/src -L$(LLAMA_LIB_DIR)/ggml/src -L$(LLAMA_LIB_DIR)/ggml/src/ggml-blas -L$(LLAMA_LIB_DIR)/common 
# Librerías del sistema y aceleración numérica
SYSTEM_LIBS =-lopenblas -lpthread -lm -lstdc++ -ldl
#Librerias para llama
LLAMA_LIBS =-lllama \
             -lggml \
             -lggml-base \
             -lggml-cpu \
             -lggml-blas

OTRAS_LIBS =-lcurl -lcjson -lxml2
 
all:LaBusqueda 
#all: EjemploUso 

EjemploUso: $(EJEMPLOS)/EjemploUso.c
	@echo "******** Compilamos en Linux EjemploUso ********************"
	$(CC) $(CCFLAGS) $(LLAMA_INC) $(GESTOR_INC) $(OTROS_INC) $(ANALISIS) $(EJEMPLOS)/EjemploUso.c $(GESTOR_SRC) $(LDFLAGS) $(LLAMA_LIBS) $(SYSTEM_LIBS) $(OTRAS_LIB) -o $(BIN)/EjemploUso.elf

	@echo ""
	@echo "****  Ejecutable Linux compilado y entregado en $(BIN)/EjemploUso.elf"
	@echo ""

LaBusqueda: $(EJEMPLOS)/LaBusqueda/LaBusqueda.c
	@echo "******** Compilamos en Linux Pruebas ********************"
	$(CC) $(CCFLAGS) $(LLAMA_INC) $(GESTOR_INC) $(OTROS_INC) $(ANALISIS) $(EJEMPLOS)/LaBusqueda/*.c $(GESTOR_SRC) $(LDFLAGS) $(LLAMA_LIBS) $(SYSTEM_LIBS) $(OTRAS_LIBS) -o $(BIN)/LaBusqueda.elf

	@echo ""
	@echo "****  Ejecutable Linux compilado y entregado en $(BIN)/Pruebas.elf"
	@echo ""

.PHONY: all 
