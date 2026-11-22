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

#define MAXTOKENS  256
#include "../includes/tiposLlamacpp.h"

/**
 * @details Carga del modelo en memoria icon el en memoria
 * 
 */
modelo_t *cargaModelo(void)
{
   // Primero definimos los parametros
   paramModelo_t parametrosModelo = llama_model_default_params();
   parametrosModelo.n_gpu_layers = 0;  // Fuerza CPU
   parametrosModelo.progress_callback = mi_progreso; // Proceso carga

   parametrosModelo.load_mode = LLAMA_LOAD_MODE_NONE; 

   // Ahora la carga del modelo en memoria.
   //modelo_t *modelo = llama_model_load_from_file(pathModelo, parametrosModelo);
   // Declaración de los símbolos generados por objcopy
   extern const char _binary_modelos_qwen2_5_coder_1_5b_instruct_q4_k_m_gguf_start[];
   extern const char _binary_modelos_qwen2_5_coder_1_5b_instruct_q4_k_m_gguf_end[];

   // Calculamos el tamaño (los símbolos son punteros, no enteros)
  size_t modelo_size = (size_t)(_binary_modelos_qwen2_5_coder_1_5b_instruct_q4_k_m_gguf_end - _binary_modelos_qwen2_5_coder_1_5b_instruct_q4_k_m_gguf_start);
   printf("Modelo detectado. Tamaño: %zu bytes (%.2f MB)\n", modelo_size, (float)modelo_size / (1024.0f * 1024.0f));
   // Carga en un fichero virtual
   FILE *model_file = fmemopen((void *)&_binary_modelos_qwen2_5_coder_1_5b_instruct_q4_k_m_gguf_start, modelo_size, "rb");
  modelo_t *modelo = llama_model_load_from_file_ptr(model_file, parametrosModelo);
  fclose(model_file);


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
   //parametrosContexto.n_ctx = 2048;  // Ventana de contexto
   //parametrosContexto.n_batch = 2048; // Tokens a procesar para llama_decodee.
   parametrosContexto.n_ctx = 512;  // Tamaño contexto
   parametrosContexto.n_batch = 512; // Tokes a procesar para llama_decodee.

   parametrosContexto.n_threads = 4;  // Hilos para generación en CPU los fisicos, GPU no poner
   parametrosContexto.n_threads_batch = 4;       // Hilos para lotes
   parametrosContexto.type_k = GGML_TYPE_Q8_0;  // a 8 bits Precisión de claves F16 en GPU
   parametrosContexto.type_v = GGML_TYPE_Q8_0;    // a 8 bits Precisión de valores F16 en GPU
   parametrosContexto.offload_kqv = false;      // true en GPU

   parametrosContexto.embeddings = false;        // No extraer embeddings
   parametrosContexto.rope_freq_base = 0.0f;     // Usar valor del modelo
   parametrosContexto.rope_freq_scale = 0.0f;    // Usar valor del modelo

   // Crear el contexto ligado al modelo
   return  llama_init_from_model(modelo, parametrosContexto);
}

/**
 * @details Crea Sampler
 * 
 * @param medir  indica si debe o no medir rendimiento
 * @return El puntero al contexto
 */
sampler_t *iniciaSampler(bool medir)
{
   paramSampler_t sparams = llama_sampler_chain_default_params();
   sparams.no_perf = !medir;

   return llama_sampler_chain_init(sparams);
}


/**
 * @details Realiza lcontrol de si debe rotar el contexto segun resultado de inferencia
 * 
 * @param contexto el contexto creado de llama.cpp
 * @param miBatch  Batch actual que nos permite conocer los tokens que se añadiran
 * @param n_past puntero a la posición actual y se actualizara si realizamos ventana de desplazamiento 
 * @param tokens_ROL  Tokens del ROL para evitar que se borren
 * @param n_token_a_borrar iTokens antiguos que se borraran
 * @return  Nada
 */
static void gestContexto(context_t *contexto, struct llama_batch_ext *miBatch, int32_t n_tokens_batch, int32_t *n_past, int32_t tokens_ROL, int32_t n_tokens_a_borrar)
{
   ////static bool errorContexto = false;
   int n_ctx = llama_n_ctx(contexto);  // Tamaño del contexto
   // Obtenemos la ultima posicion ocupada
   llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(contexto), 0);
   int n_ctx_usada = (max_pos == -1) ? 0 : max_pos + 1;

   //printf("Tamaño contexto y el usado %d -- %d y los token del batch %d\n",n_ctx,n_ctx_usada,miBatch.n_tokens);

   if (n_ctx_usada + n_tokens_batch > n_ctx)   // Mira si al añadir el nuevo Batch se excede el contexto
   {
       printf("Límite de contexto alcanzado (%d/%d). Aplicando Ventana Deslizante... %d\n", n_ctx_usada, n_ctx,n_tokens_a_borrar);
        
       /* 
       // Opción Limpieza total y llamada como si empezaramos.
       printf("************ Se limpia total y se empieza*******\n");
       llama_memory_seq_rm(llama_get_memory(contexto), 0, -1, -1);
       free(prompt_formateado);  // Limpia volvemos a inferir
       inferencia(rolSystem, prompt, modelo, contexto, miSampler,vocab);
       return;
       */ 

       // Técnica avanzada de "Sliding Window" (Ventana Deslizable).
       // Límites de la "poda" p0(Despues del ROL) p1 + borrado
       int32_t p0 = tokens_ROL;   //Fin ROL
       int32_t p1 = tokens_ROL + n_tokens_a_borrar;
       llama_memory_seq_rm(llama_get_memory(contexto), 0, p0, p1); // Elimina bloque antiguo

       // Movemos lo que quedó al final hacia atrás para cerrar el hueco asi tendremos la memoria contigua
       llama_memory_seq_add(llama_get_memory(contexto), 0, p1, -1, -n_tokens_a_borrar); 

       // CRITICO: Recalcular n_past para que coincida con la nueva realidad del KV Cache
       // Tras el desplazamiento, la nueva máxima posición es la antigua menos el desplazamiento.
       *n_past = llama_memory_seq_pos_max(llama_get_memory(contexto), 0) + 1;
        
       printf("Ventana deslizante aplicada con éxito. Nueva posición n_past: %d\n", *n_past);
   }
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
void inferencia(const char *rolSystem, const char *prompt, modelo_t *modelo, context_t *contexto, sampler_t *miSampler, const vocab_t *vocab, char *respuesta, int tamRespuesta)
{
   int posRespuesta = 0; // Para rellenar respuesta.
   respuesta[posRespuesta] = 0; // Fin de la respuesta, inicia en blanco

   mensajeChat_t rol_System[] = {{"system", rolSystem }};    // define el rol de como comportarse.
   mensajeChat_t rol_User[] = {{"user",   prompt}};          // La pregunta
   // Calculamos los tokens del ROL
   // Mide Bytes solo del ROL por eso n_mensajes=1
   int32_t bytes_ROL = llama_chat_apply_template(llama_model_chat_template(modelo, NULL), rol_System, 1, false, NULL, 0);  //No se añade el encabezado de asistente.

   // Buffer temporal para el ROL formateado
   char * prompt_ROL = malloc(bytes_ROL + 1);
   llama_chat_apply_template(llama_model_chat_template(modelo, NULL), rol_System, 1, false, prompt_ROL, bytes_ROL);
   prompt_ROL[bytes_ROL] = '\0';

   // Contamos TOKENS usamos NULL en el buffer de tokens y 0 solo cuenta
   int32_t tokens_ROL = llama_tokenize(vocab, prompt_ROL, bytes_ROL, NULL, 0, true, true ) * -1 ; // add_especial es true porque el ROL abre la secuencia
   free(prompt_ROL); // Limpiamos memoria temporal del ROL que no sera necesaria

   // **** Si es el primer mensaje se añade tanto el rol system como el rol_user, en los siguientes solo User
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
   // **********  6.1  Generamos el Prompt Formateado con el mensaje
   const char * tmpl = llama_model_chat_template(modelo, NULL);  // Plantilla
   int32_t bytes_prompt_final = llama_chat_apply_template(tmpl, mensajes, n_mensajes, true, NULL, 0);  // calculo del buffer necesario
   char * prompt_formateado = malloc(bytes_prompt_final + 1);
   llama_chat_apply_template(tmpl, mensajes, n_mensajes, true, prompt_formateado, bytes_prompt_final);  // Aplica
   prompt_formateado[bytes_prompt_final] = '\0'; // Asegurar cierre C

   // ********** 6.2  Tokenizamos el prompt formateado con mensaje
   // Cuando pedimos sin buffer de salida por logica no tiene espacio, así que dara un negativo.
   int32_t tokens_necesarios_negativos = llama_tokenize(vocab,prompt_formateado,strlen(prompt_formateado),NULL,0,elPrimero,true);
   int32_t n_prompt = tokens_necesarios_negativos * -1;
   llama_token *prompt_tokens = (llama_token*) malloc(n_prompt*sizeof(llama_token));
   if (llama_tokenize(vocab,prompt_formateado,strlen(prompt_formateado),prompt_tokens,n_prompt,elPrimero,true)<0 ) { fprintf(stderr,"%s: Fallo al tokenizar\n",__func__); exit(-133) ; }
   free(prompt_formateado); // Ya no es neceserio por lo tanto lo borramos cuanto antes.

   // ******** 6.3  Preparando el Batch de la inferencia
   // Construir el batch1 respetando la posición actual del KV Cache
   int32_t start_pos = 0;
   if (!elPrimero) 
   {
       llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(contexto), 0);
       start_pos = (max_pos == -1) ? 0 : max_pos + 1;
   }
   // Creamos el batch1 manualmente para controlar las posiciones 'pos'

   struct llama_batch_ext *miBatch1 = llama_batch_ext_init(contexto);
   int n_tokens = n_prompt;

   llama_batch_ext_clear(miBatch1); // Limpiamos por seguridad
   //   Llenamos el Batch TOKEN por TOKEN prompt_tokensn_prompt posicion 0
   for (int i = 0; i < n_prompt; i++) 
   {
      // Token a la secuencia 0 y obtenemos idx que es la posicion interna asignada
      int32_t idx = llama_batch_ext_add_token(miBatch1, 0, prompt_tokens[i]);
      // Establecemos la posición (pos) para la continuidad del caché KV
      llama_pos pos = start_pos + i;
      llama_batch_ext_set_pos(miBatch1, idx, &pos);
   }
   // Activamos los logit solo para el ultimo
   llama_batch_ext_set_output_logits(miBatch1, n_prompt - 1, true);

   if (llama_process(contexto, LLAMA_PROCESS_TYPE_DECODE, miBatch1) != 0) { fprintf(stderr, "Error crítico: falló el decode del prompt inicial.\n"); llama_batch_ext_free(miBatch1); free(prompt_tokens); exit(-198); }
   llama_batch_ext_free(miBatch1); // Liberamos el batch de prompt tras su uso
   free(prompt_tokens);

   // Variables de control del bucle
   int n_predict = MAXTOKENS;          // Máximo de tokens a generar
   llama_token ID_nuevoToken;
   int n_generated = 0;          // Contador de tokens generados
   int n_past = start_pos + n_tokens;        // Posición absoluta actual en la secuencia (KV Cache, contexto)

   // Inicializar el batch de generación UNA SOLA VEZ fuera del bucle
   struct llama_batch_ext *miBatch_Sig = llama_batch_ext_init(contexto);
   n_tokens = 1;

   //  Bucle de generación autoregresivao
   for (; n_generated < n_predict; ++n_generated) 
   {
      // A. Muestreo del siguiente token (usa los logits del último token evaluado, índice -1)
      ID_nuevoToken = llama_sampler_sample(miSampler, contexto, -1);

      // B. Condición de parada: Fin de Generación (EOS / EOT) - API ACTUALIZADA
      if (llama_vocab_is_eog(vocab, ID_nuevoToken)) { break; }

      // C. Destokenización convertimos TOKENs a Texto
      char buf[MAXTOKENS];
      // Usamos el modelo para obtener el vocabulario internamente en token_to_piece
      int n = llama_token_to_piece(vocab, ID_nuevoToken, buf, sizeof(buf), 0, true);
      if (n < 0) { fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__); respuesta[posRespuesta] = 0; exit(-102) ; } // La respuesta sera parcial
      // Vamos rellenado el buffer de respuesta
      for (int i=0; i<n; i++) { respuesta[posRespuesta] = buf[i]; posRespuesta++; }

       // D. Preparar el batch para el SIGUIENTE token generado y lo decodifica, cambia solo datos que se han podido mover
       llama_batch_ext_clear(miBatch_Sig);  // Limpia batch anterior manteniendolo en memoria.
       int32_t idx = llama_batch_ext_add_token(miBatch_Sig, 0, ID_nuevoToken);  // Nuevo token a seq 0
       llama_pos pos = n_past;
       llama_batch_ext_set_pos(miBatch_Sig, idx, &pos);
       llama_batch_ext_set_output_logits(miBatch_Sig, idx, true);

       if (llama_process(contexto, LLAMA_PROCESS_TYPE_DECODE, miBatch_Sig) != 0) { fprintf(stderr, "Error crítico: falló el decode del token generado.\n"); llama_batch_ext_free(miBatch_Sig); exit(-103); }

       // E. Actualizar la posición absoluta para la próxima iteración
       n_past++ ;

       // F.   Analizamos el espacio del contexto
       gestContexto(contexto, miBatch_Sig, 1, &n_past, tokens_ROL,256 ) ;

   }  // Fin del bucle de inferencia
   respuesta[posRespuesta] = 0; // Fin de la respuesta
   llama_batch_ext_free(miBatch_Sig); // Liberamos el batch temporal
}


#endif
