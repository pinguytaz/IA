/*********************************************************************************
 * Funciones para las carga de modelo, sampler, etc
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 ********************************************************************************/
#ifndef FUNCIA_H
#include "llama.h"
#define FUNCIA_H

#include "../includes/tiposLlamacpp.h"

/**
 * @details Carga del modelo en memoria
 * 
 * @param Camino al modelo de memoria
 * @return El puntero al modelo
 */
modelo_t *cargaModelo(const char* pathModelo)
{
   // Primero definimos los parametros
   paramModelo_t parametrosModelo = llama_model_default_params();
   parametrosModelo.n_gpu_layers = 0;  // Fuerza CPU
   parametrosModelo.use_mmap  = true; // Optimiza memoria
   parametrosModelo.use_mlock = false; //No se envia a SWAP
   parametrosModelo.progress_callback = mi_progreso; // Proceso carga

   // Ahora la carga del modelo en memoria.
   modelo_t *modelo = llama_model_load_from_file(pathModelo, parametrosModelo);

   return modelo;
}


/**
 * @details Inicia el contexto que se liga al modelo
 * 
 * @param modelo
 * @return El puntero al contexto
 */
context_t *iniciaContexto(modelo_t *modelo)
{
   paramContext_t parametrosContexto = llama_context_default_params();
   // Optimización para OpenBLAS en CPU
   //parametrosContexto.n_ctx = 2048;  // Tamaño contexto
   //parametrosContexto.n_batch = 2048; // Tokes a procesar para llama_decodee.
   parametrosContexto.n_ctx = 512;  // Tamaño contexto
   parametrosContexto.n_batch = 512; // Tokes a procesar para llama_decodee.

   // Crear el contexto ligado al modelo
   return  llama_init_from_model(modelo, parametrosContexto);
}

/**
 * @details Crea Sampler, muy estricto para 
 * 
 * @param modelo
 * @return El puntero al contexto
 */
sampler_t *iniciaSampler(void)
{
   paramSampler_t sparams = llama_sampler_chain_default_params();
   sparams.no_perf = false;

   return llama_sampler_chain_init(sparams);
}


/**
 * @details Realiza la inferencia del prompt pasado
 * 
 * @param Rol del sistema 
 * @param Mensaje
 * @param modelo
 * @param contexto del sistema 
 * @param Sampler para saber que palabras seleccionar
 * @param Vocabulario evitamos cargrlo cada vez
 * @param buffer de respuesta
 * @param Maximo temaño respuesta
 * @return  Nada
 */
void inferencia(const char *rolSystem, const char *prompt, modelo_t *modelo,
                context_t *contexto, sampler_t *miSampler, const vocab_t *vocab,
		char *respuesta, int tamRespuesta)
{
   static bool errorContexto = false;

   int posRespuesta = 0; // Para rellenar respuesta.
   respuesta[posRespuesta] = 0; // Fin de la respuesta, inicia en blanco

   mensajeChat_t rol_System[] = {{"system", rolSystem }};
   mensajeChat_t rol_User[] = {{"user",   prompt}};

    // Calculamos los totkes del ROL, por si debemos rotar contexto.
    // Mide Bytes solo del ROL por eso n_mensajes=1
    int32_t bytes_ROL = llama_chat_apply_template(llama_model_chat_template(modelo, NULL), rol_System, 1, false, NULL, 0);  //No se añade el encabezado de asistente.
    // Buffer temporal para el ROL formateado
    char * prompt_ROL = malloc(bytes_ROL + 1);
    llama_chat_apply_template(llama_model_chat_template(modelo, NULL), rol_System, 1, false, prompt_ROL, bytes_ROL); 
    prompt_ROL[bytes_ROL] = '\0';
    // Contamos TOKENS asamos NULL en el buffer de tokens y 0 solo cuenta
    int32_t tokens_ROL = llama_tokenize( vocab, prompt_ROL, bytes_ROL, NULL, 0, true, true ); // add_especial es true porque el ROL abre la secuencia
    free(prompt_ROL); // Limpiamos memoria temporal del ROL que no sera necesaria

    // Vemos si es el primer mensaje para el contexto.
    bool elPrimero = llama_memory_seq_pos_max(llama_get_memory(contexto), 0) == -1;
    int n_mensajes ;
    mensajeChat_t mensajes[2];  // Preparado para ROL y User
    if(elPrimero)
    {
       //printf("********************* La primera Vez\n");
       mensajes[0] = rol_System[0];
       mensajes[1] = rol_User[0];
       n_mensajes = 2; // Tenemos dos mensajes en el historial
    }
    else
    {
       //printf("***************** Siguientes\n");
       mensajes[0] = rol_User[0];
       n_mensajes = 1; // Tenemos 1
    }
    //printf("MSG0 ROL:%s \nContenido:%s\n",mensajes[0].role,mensajes[0].content);
    //printf("MSG1 ROL:%s \nContenido:%s\n",mensajes[1].role,mensajes[1].content);

    //printf("\n**** Inferencia de %s es la primera vez: %d ************\n",prompt,elPrimero);
    // Generamos el Prompt Formateado con el mensaje
    int32_t bytes_prompt_final = llama_chat_apply_template(llama_model_chat_template(modelo, NULL), mensajes, n_mensajes, true, NULL, 0);
    char * prompt_formateado = malloc(bytes_prompt_final + 1);
    llama_chat_apply_template(llama_model_chat_template(modelo, NULL), mensajes, n_mensajes, true, prompt_formateado, bytes_prompt_final);
    prompt_formateado[bytes_prompt_final] = '\0'; // Asegurar cierre C

    // Tokenizamos el prompt formateado con mensaje
    int32_t tokens_necesarios_negativos = llama_tokenize(vocab,prompt_formateado,strlen(prompt_formateado),NULL,0,elPrimero,true);
    int32_t n_tokens = tokens_necesarios_negativos *  -1;
    llama_token *prompt_tokens = (llama_token*) malloc(n_tokens*sizeof(llama_token));
    if (llama_tokenize(vocab,prompt_formateado,strlen(prompt_formateado),prompt_tokens,n_tokens,elPrimero,true)<0 )
    {
        fprintf(stderr,"%s: Fallo al tokenizar\n",__func__);
        return ;
    }

    // Preparando el Batch de la inferencia
    llama_batch miBatch = llama_batch_get_one(prompt_tokens, n_tokens);

    llama_token ID_nuevoToken;
    int n_predict = 256;
    for (int n_pos = 0; n_pos + miBatch.n_tokens < n_tokens + n_predict; )
    {
        // Analizamos el espacio del contexto
        int n_ctx = llama_n_ctx(contexto);
        int n_ctx_usada = llama_memory_seq_pos_max(llama_get_memory(contexto), 0) + 1;
	//printf("Tamaño contexto y el usado %d -- %d y los token del batch %d\n",n_ctx,n_ctx_usada,miBatch.n_tokens);
        if (n_ctx_usada + miBatch.n_tokens > n_ctx) 
	{
	   if(errorContexto)
	   {
               fprintf(stderr, "ERROR Tamaño de contexto %d y petición de %d\n",n_ctx,n_ctx_usada + miBatch.n_tokens); 
               exit(0);
           }
	   else
	   {
	      errorContexto = true; // No limpiamos dos veces seguidas.

	     /* 
	      // Opción Limpieza total y llamada como si empezaramos.
	      printf("************ Se limpia total y se empieza*******\n");
	      llama_memory_seq_rm(llama_get_memory(contexto), 0, -1, -1);
              free(prompt_formateado);  // Limpia volvemos a inferir
              inferencia(rolSystem, prompt, modelo, contexto, miSampler,vocab);
	      return;
	     */ 

	      // Técnica avanzada de "Sliding Window" (Ventana Deslizable).
              // Definimos bloque a borrar
	      int32_t n_tokens_a_borrar = 256;
	      printf("************ Deslizamos %d bloques ***\n",n_tokens_a_borrar);
	      // Límites de la "poda" p0(Despues del ROL) p1 + borrado
              int32_t p0 = tokens_ROL;   //Fin ROL
              int32_t p1 = tokens_ROL + n_tokens_a_borrar;
              llama_memory_seq_rm(llama_get_memory(contexto), 0, p0, p1);

	      // Movemos lo que quedó al final hacia atrás para cerrar el hueco
              // Asi tendremos la memoria contigua
              llama_memory_seq_add(llama_get_memory(contexto), 0, p1, -1, 
	                           -n_tokens_a_borrar);
	     }
        }
	else { errorContexto = false; }   // Fue bien iniciar errorContexto.

	// Lanzamos prompt y contexto para inferir
        if (llama_decode(contexto, miBatch) != 0)
        {
            fprintf(stderr, "Error crítico: falló el decode del prompt inicial.\n");
            return ;
        }
        n_pos += miBatch.n_tokens;

        ID_nuevoToken = llama_sampler_sample(miSampler, contexto, -1);
        // Esto es el fin de la generación
        if (llama_vocab_is_eog(vocab, ID_nuevoToken))
        {
            break;
        }

        // Convierte el TOKEN a String
        char buf[256];
        int n = llama_token_to_piece(vocab, ID_nuevoToken, buf, sizeof(buf), 0, true);
        if (n < 0)
        {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            respuesta[posRespuesta] = 0; // Fin de la respuesta sera parcial
            return ;
        }
	// Vamos rellenado el buffer de respuesta
	for (int i=0; i<n; i++)
	{
	   respuesta[posRespuesta] = buf[i];
	   posRespuesta++;
	}
        //buf[n] = 0;
        //printf("%s", buf);

        // Prepara el sigiente lote con TOKEN sample
        miBatch = llama_batch_get_one(&ID_nuevoToken, 1);
    }
    respuesta[posRespuesta] = 0; // Fin de la respuesta

    // Libera las memorias auxiliares utilizadas
    free(prompt_formateado);
}


#endif
