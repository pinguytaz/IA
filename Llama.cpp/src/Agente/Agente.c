/***********************************************************************************
 * Agente	Desarrollo de un sistema de agentes con llama.cpp
 *		El sistema es bastante simple y se basa en el realizado en python
 *		para la presentación de introducción Agente IA.
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 *	Utilizaremos el mismo modelo que el usado en python, pero gguf.
 *	qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
 *
 * librerias:
 *	llama.cpp	Libreria que nos permite ganerar inferencia LLM en C
 ***********************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "llama.h"

#include "../includes/tiposLlamacpp.h"
#include "../includes/funcCallB.h"
#include "funcIA.h"
#include "funcAgentes.h"

int main(int argc, char ** argv)
{
   if(argc != 1)
   {
      printf("Uso: %s \n",argv[0]);
      return -1;
   }


    printf("Inicializando sistema Agentes...\n");
    llama_log_set(mi_funcion_log_silenciosa, NULL);  // Silenciamos LOG

    /************ Iniciamos Backend *************/
    llama_backend_init();
    /************ Cargamos modelo de Memoria para el agente *************/
    const char *pathModelo="./modelos/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    modelo_t *modelo = cargaModelo(pathModelo);
    if (modelo == NULL) 
    {
        fprintf(stderr,"Error al cargar el modelo%s\n",pathModelo);
	return -2;
    }
    char nombreModelo[64];
    int32_t res = llama_model_meta_val_str(modelo, "general.name", nombreModelo, sizeof(nombreModelo));
    if (res > 0) printf("\nModelo cargado %s\nFichero: %s\n",nombreModelo,pathModelo);


    /************ Configuramos el Contexto ****************************/
    context_t *contexto = iniciaContexto(modelo);
    if (contexto == NULL)
    {
        fprintf(stderr, "ERROR: No se pudo crear el contexto de llama.\n");
        return 1;
    }

    /******** Crea Sampler **************************************/
    sampler_t *miSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(miSampler, llama_sampler_init_greedy()); // Mas exacta

    /************* Cargams el vocabulario ********/
    const vocab_t *vocab = llama_model_get_vocab(modelo);

    /************** Obtenemos el Rol de System ****************/
    char rolSystem[1024]; // Rol del sistema Agentico.
    FILE *f = fopen("./roles/RolAgente.txt", "rb");
    if (!f) return -3;
    fseek(f, 0, SEEK_END);  // Al final para ver tamaño
    long tam = ftell(f);  // Tamaño del fichero
    rewind(f);  // Volvemos al inicio
    if(tam > 1024) return -4;
    size_t leidos = fread(rolSystem, 1, (size_t)tam, f);
    rolSystem[leidos] = '\0';
    fclose(f);

    char prompt[256];
    char respuesta[512];
    while(true)
    {
        printf("Usuario> ");
        if (fgets(prompt, sizeof(prompt), stdin) == NULL) { break; }

        if (strchr(prompt, '\n') == NULL) 
	{
            printf("Error: entrada demasiado larga\n");
            int c;
            while ((c = getchar()) != '\n' && c != EOF) {;}
	    continue;
        }
        prompt[strcspn(prompt, "\n")] = '\0';
	if(strcmp(prompt,"Salir")==0) 
	{
	   printf("Recibida orden de cerrar el agente\n");
	   break;
        }

	inferencia(rolSystem, prompt,modelo, contexto, miSampler, vocab, respuesta,sizeof(respuesta));
	/*
	int32_t min_pos = llama_memory_seq_pos_min(llama_get_memory(contexto), 0);
        int32_t max_pos = llama_memory_seq_pos_max(llama_get_memory(contexto), 0);
        printf("DEBUG: Memoria activa desde pos %d hasta %d (Total: %d tokens)\n",
               min_pos, max_pos, max_pos - min_pos + 1);
	*/
	printf("\nIA>: %s\n",respuesta);
	// Obtenemos funciones y realiza llamadas.
	procesarRespuestas((const char *)respuesta);

    }
    /************ Limpieza de memoria *************/
    llama_sampler_free(miSampler);
    llama_free(contexto);
    llama_model_free(modelo);
    llama_backend_free(); 

    printf("***************** ¡Fin programa Agentes! ******************\n"); 
    return 0;
}

