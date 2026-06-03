# Rutas y flags
CC = gcc
CC_W = x86_64-w64-mingw32-gcc
BIN = ./bin
SRC = ./src
LLAMA_DIR = ./llama.cpp
LLAMA_LIB_DIR = $(LLAMA_DIR)/build
LLAMA_LIB_DIR_W = $(LLAMA_DIR)/buildWindows

# Rutas cabecera y librerias
LLAMA_INC = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/common

CCFLAGS = -O2 -Wall -fopenmp
LDFLAGS = -L$(LLAMA_LIB_DIR)/src -L$(LLAMA_LIB_DIR)/ggml/src -L$(LLAMA_LIB_DIR)/ggml/src/ggml-blas -L$(LLAMA_LIB_DIR)/common 
LDFLAGS_W = -L$(LLAMA_LIB_DIR_W)/src -L$(LLAMA_LIB_DIR_W)/ggml/src -L$(LLAMA_LIB_DIR_W)/ggml/src/ggml-blas -L$(LLAMA_LIB_DIR_W)/common 
# Librerías del sistema y aceleración numérica
SYSTEM_LIBS = -lopenblas -lpthread -lm -lstdc++
#SYSTEM_LIBS_W = -lws2_32 -lbcrypt -luser32 -lpthread -lm -lstdc++
SYSTEM_LIBS_W = -lpthread -lm -lstdc++
#Librerias para llama
LLAMA_LIBS = -lllama \
             -lggml \
             -lggml-base \
             -lggml-cpu \
             -lggml-blas 

LLAMA_LIBS_W = -lllama \
             -l:ggml.a \
             -l:ggml-base.a \
             -l:ggml-cpu.a 

all: Agente

Base: $(SRC)/Base/Base.c 
	@echo "******** Compilamos en Linux Base ********************"
	$(CC) $(CCFLAGS) $(LLAMA_INC) $(SRC)/Base/Base.c $(LDFLAGS) $(LLAMA_LIBS) $(SYSTEM_LIBS) -ldl -o $(BIN)/Base.elf
	@echo ""
	@echo "****  Ejecutable Linux compilado y entregado en $(BIN)/Base.elf"
	@echo ""
	@echo "******** Compilamos en Windows  Base ********************"
	$(CC_W) $(CCFLAGS) -static $(LLAMA_INC) $(SRC)/Base/Base.c $(LDFLAGS_W) $(LLAMA_LIBS_W) $(SYSTEM_LIBS_W) -o $(BIN)/Base.exe
	@echo ""
	@echo "***  Ejecutable Windows compilado y entregado en $(BIN)/Base.exe"
	@echo ""

Agente: $(SRC)/Agente/Agente.c 
	@echo "******** Compilamos en Linux Agente ********************"
	$(CC) $(CCFLAGS) $(LLAMA_INC) $(SRC)/Agente/Agente.c $(LDFLAGS) $(LLAMA_LIBS) $(SYSTEM_LIBS) -ldl -o $(BIN)/Agente.elf
	@echo ""
	@echo "****  Ejecutable Linux compilado y entregado en $(BIN)/Agente.elf"
	@echo ""

.PHONY: all Base Agente
