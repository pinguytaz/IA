/***********************************************************************************
 * Agente2	Version de agente con el modelo en memoria
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
   if(argc != 1) { printf("Uso: %s \n",argv[0]); return -1; }


   printf("Inicializando sistema Agentes...\n");
   llama_log_set(mi_funcion_log_silenciosa, NULL);  // Silenciamos LOG

   /************ 1.- Iniciamos Backend *************/
   llama_backend_init();

   /************ 2.- Cargamos modelo de Memoria para el agente *************/
   //const char *pathModelo="./modelos/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
   //const char *pathModelo="./modelos/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf";

   modelo_t *modelo = cargaModelo();     //pathModelo);
   if (modelo == NULL) { fprintf(stderr,"Error al cargar el modelo\n"); return -2; }
   char nombreModelo[64];
   int32_t res = llama_model_meta_val_str(modelo, "general.name", nombreModelo, sizeof(nombreModelo));
   if (res > 0) printf("\nModelo cargado %s\n",nombreModelo);

   /************ 3.- Configuramos el Contexto y la memoria conversacional ****************************/
   context_t *contexto = iniciaContexto(modelo);
   if (contexto == NULL) { fprintf(stderr, "ERROR: No se pudo crear el contexto de llama.\n"); return -3; }

   /******** 4.- Crea Sampler **************************************/
   sampler_t *miSampler = iniciaSampler(false);   // poner true si deseamos info de metricas sampler
   // Montamos las cadenas de Sampler.
   llama_sampler_chain_add(miSampler, llama_sampler_init_greedy()); // Mas exacta
   //llama_sampler_chain_add(miSampler, llama_sampler_init_top_k(3));
   //llama_sampler_chain_add(miSampler, llama_sampler_init_top_p(0.95, 1));
   //llama_sampler_chain_add(miSampler, llama_sampler_init_temp(0.25));
   //llama_sampler_chain_add(miSampler, llama_sampler_init_dist(42)); // Semilla fija

   /************* 5.- Cargamos el vocabulario ********/
   const vocab_t *vocab = llama_model_get_vocab(modelo);

   /************** 6.- Inferencia  ****************/

   /******* Cargamos el ROL del sistema definido en un fichero. ****/
   char rolSystem[1536]; // Rol del sistema Agentico.
   FILE *f = fopen("./roles/RolAgente.txt", "rb");
   if (!f) return -3;
   fseek(f, 0, SEEK_END);  // Al final para ver tamaño
   long tam = ftell(f);  // Tamaño del fichero
   rewind(f);  // Volvemos al inicio
   if(tam > 1536) { fprintf(stderr,"El ROL esta pasandose de tamaño\n"); exit(-4); }
   size_t leidos = fread(rolSystem, 1, (size_t)tam, f);
   rolSystem[leidos] = '\0';
   fclose(f);

   char prompt[256];
   char respuesta[512];
   while(true)     // Bucle sin realizando preguntas, hasta que le digamos "Salir"
   {
       // *** Recogemos la pregunta y si es "Salir" salimos del bucle y finalizamos
       printf("Usuario> ");
       if (fgets(prompt, sizeof(prompt), stdin) == NULL) { break; }

       if (strchr(prompt, '\n') == NULL) 
       {
           printf("Error: entrada demasiado larga\n");
           int c;
           while ((c = getchar()) != '\n' && c != EOF) {;} 
	   continue;
       }
       prompt[strcspn(prompt, "\n")] = '\0'; // Limpia retorno de linea
       if(strcmp(prompt,"Salir")==0)  { printf("Recibida orden de cerrar el agente\n"); break; }

       // *** Realiza la inferencia
       inferencia(rolSystem, prompt,modelo, contexto, miSampler, vocab, respuesta,sizeof(respuesta));
       printf("\nIA>: %s\n",respuesta);     // Imprimimos la respuesta dada por la inferencia

       // Obtenemos funciones y realiza llamadas.
       procesarRespuestas((const char *)respuesta);  // Segun la respuesta vemos y ejecutamos las funciones correspondientes.


       //**********************+ Metricas para desarrollo
       /* Metricas del Sampler solo si le definimos como true (no_pref=false)
       printf("\n=== MÉTRICAS DE SAMPLING ===\n");
       struct llama_perf_sampler_data data = llama_perf_sampler(miSampler);
       printf("\nDatos detallados:\n");
       printf("  Tiempo total de sampling: %.2f ms\n", data.t_sample_ms);
       printf("  Tokens muestreados: %d\n", data.n_sample);
       if (data.n_sample > 0) 
       {
           printf("  Tiempo promedio por token: %.4f ms\n", data.t_sample_ms / data.n_sample);
           printf("  Tokens por segundo: %.2f\n", 1000.0 * data.n_sample / data.t_sample_ms);
       }
       */

       /* 
	int32_t min_pos = llama_memory_seq_pos_min(llama_get_memory(contexto), 0);
        int32_t max_pos = llama_memory_seq_pos_max(llama_get_memory(contexto), 0);
        printf("DEBUG: Memoria activa desde pos %d hasta %d (Total: %d tokens)\n",
               min_pos, max_pos, max_pos - min_pos + 1);
        */
        //********************** Fin Metricas para desarrollo

   }  // Finaliza el bucle sin fin hasta "Salir"

   /************ 7.- Limpieza de memoria *************/
   llama_sampler_free(miSampler);
   llama_free(contexto);
   llama_model_free(modelo);
   llama_backend_free(); 

   printf("***************** ¡Fin programa Agentes! ******************\n"); 
   return 0;
}

