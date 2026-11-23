/*********************************************************************************
 * Gestor IA con llama.cpp
 *   Marco para realizar proyectos de IA con llama.cpp, permitiendo la carga de 
 *   modelos a la vez, y cambiar datos al realizar la inferencia.
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 ********************************************************************************/
#ifndef GESTORLLAMACPP_H
#include "llama.h"
#include "tiposLlamacpp.h"
#include "funciones.h"
#define GESTORLLAMACPP_H

#define MAX_MODELOS 5        // Maximo numero de modelos en memoria
#define MAXTOKENS  256       // Valor po defecto de TOKENs a leer
#define CHUNK_SIZE 8192      // 8 KB por expansión

///////////////////////  ESTRUCTURAS Y TIPOS ///////////////////////////////////////
// *****  Información de modelo, guardado en lista enlazada para borrar y liberar espacio
typedef struct
{
    int id;    // Identificador numerico del modelo que generamos nosotros.
    char nombre[64];           // Nombre identificador del modelo
    char pathModelo[256];    // Ruta al archivo .gguf
    modelo_t *modelo;       // Puntero al modelo cargado
    context_t *contexto;       // Puntero al contexto
    paramContext_t params_ctx;  // Se guardan los parametros de contexto actuales
} ModeloInfo;

typedef struct Nodo
{
    ModeloInfo dato;      
    struct Nodo *siguiente;   // puntero al siguiente nodo 
} Nodo;

typedef struct 
{
    Nodo *cabeza;
    int count; 
    int modelos;    // Numero de modelos cargados numca mas de MAX_MODELOS 
} GestorModelos;

// ******     Estrategias de limpieza
typedef enum 
{
    COMPLETO,              // Limpia por completo la caché KV (sin histórico)
    VENTANA_DESLIZANTE  // Mantiene una ventana deslizante de tokens (histórico limitado)
} LimpiaContexto_t;
///////////////////////////////////////////////////////////////////////////////////


///////////////////////  FUNCIONES PUBLICAS Uso del Gestor ///////////////////////////////////////

void iniciaGestor(bool log);   		// Inicia Gestor
void destructorGestor(void);   		// Destruye gestor
int cargaModelo(const char *nombre, char* pathModelo);  // Carga modelo en el gestor.
void limpiaGestor(void);     		// Elimina modelos del gestor pero continua backend.
void listaModelos(void);    // Lista los modelos cargados
int modelosCargados(void);  // Número de modelos cargados

int32_t vocab_n_tokens(int id);         // Nos el total de tokens del vocabulario par Sampler
int borraModelo(int id);      		// Borra modelo indicado
char *get_NombreModelo(int indice);	// Obtiene nombre Modelo cargado
char *get_NombreArchivo(int indice);	// Obtienen el nombre del fichero del modelo cargado.
//modelo_t *get_Modelo(int indice);	// Nos da el modelo (formato llama
paramContext_t get_ParamContexto(int id);   // Parametros de contexto del modelo.
int put_ParamContexto(int id, paramContext_t parametros); // Pone parametros de contexto

int inferir(int id,  const char *pregunta, char *rolSystem, sampler_t *sampler, StringDinamico_t *respuesta, 
            bool limpio, int limTokens, LimpiaContexto_t estrategia);

#endif

