#include <stdlib.h>
#include <string.h>
#include "GestorLlamaCPP.h"

static GestorModelos gestor = {0};  // Variable que contiene la información de los modelos cargados
#define CHUNK_SIZE 8192 // 8 KB por expansión

////////////////////////////////////////////////  FUNCIONES PRIVADAS (static)  //////////////////////////////////
/*************** Silencia LOG******************************************/
static void mi_funcion_log_silenciosa(enum ggml_log_level level, const char * text, void * user_data)
{
   // No hacemos nada aquí. Al dejarla vacía, bloqueamos el texto.
}

/*************** Visualiza el proceso de carga del modelo *******************************/
static bool mi_progreso(float progreso, void *datosUsuario)
{
    printf("\rCargando modelo: %.1f%% ", progreso * 100);
    fflush(stdout); // Para que se vea en tiempo real
    return true; // Continuar cargando
}

/************** Limpia el nodo pasado como parametro *********/
static void limpiaModelo(ModeloInfo *dato)   // Borra un modelo
{ 
   if (dato != NULL) 
   { 
      llama_free(dato->contexto);
      llama_model_free(dato->modelo); 
   } 
}

/*************** Busca modelo con el identificador pasado como parametro ***/
static Nodo *buscaNodo(int id)
{
   if (gestor.cabeza == NULL) { return NULL; }   // Gestor sin modelos

   Nodo *actual = gestor.cabeza;

   // Buscar el nodo
   while (actual != NULL && actual->dato.id != id) { actual = actual->siguiente; }

   if (actual == NULL) { return NULL; }  // no se encontró

   return actual;
}

/************* Tokeniza prompt RolSystem + Usuario(pregunta)  *******/
static int32_t calculaTokensROL(mensajeChat_t rol_User[], mensajeChat_t rol_System[], modelo_t *modelo, vocab_t *vocabulario)
{

  // Calculamos los tokens del ROL
  // Mide Bytes solo del ROL por eso n_mensajes=1
  int32_t bytes_ROL = llama_chat_apply_template(llama_model_chat_template(modelo, NULL), rol_System, 1, false, NULL, 0);  //No se añade el encabezado de asistente.

  // Buffer temporal para el ROL formateado
  char * prompt_ROL = malloc(bytes_ROL + 1);
  llama_chat_apply_template(llama_model_chat_template(modelo, NULL), rol_System, 1, false, prompt_ROL, bytes_ROL);
  prompt_ROL[bytes_ROL] = '\0';

  // Contamos TOKENS usamos NULL en el buffer de tokens y 0 solo cuenta
  int32_t n_sys_tokens = llama_tokenize(vocabulario, prompt_ROL, bytes_ROL, NULL, 0, true, true ) * -1 ; // add_especial es true porque el ROL abre la secuencia
  free(prompt_ROL); // Limpiamos memoria temporal del ROL que no sera necesaria


   return n_sys_tokens;
}

/******
 * Realiza control de si debe rotar el contexto segun resultado de inferencia
 *
 * Parametros:
 * 	contexto:		El contexto creado de llama.cpp
 * 	miBatch  		Batch actual que nos permite conocer los tokens que se añadiran
 * 	n_tokens_batch 		Tokens del batch
 * 	n_past 			puntero a la posición actual y se actualizara si realizamos ventana de desplazamiento 
 * 	n_sys_tokens  		Tokens del ROL para evitar que se borren .
 *      n_token_a_borrar	Tokens antiguos que se borraran en ventana desplazamiento
 *      rolSystem	        prompt de systema con el ROL a tomar
 *      estrategia	        Estrategia de limpieza del contextor
 *
 *******/
static void gestContexto(context_t *contexto, batch_ext_t *miBatch, int32_t n_tokens_batch, int32_t *n_past, int32_t n_sys_tokens, 
                         int32_t n_tokens_a_borrar, char *rolSystem, LimpiaContexto_t estrategia)
{
   int n_ctx = llama_n_ctx(contexto);  // Tamaño del contexto

   // Obtenemos la ultima posicion ocupada
   llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(contexto), 0);
   int n_ctx_usada = (max_pos == -1) ? 0 : max_pos + 1;

   if (n_ctx_usada + n_tokens_batch > n_ctx)   // Mira si al añadir el nuevo Batch se excede el contexto
   {
      printf("Límite de contexto alcanzado (%d/%d). \n", n_ctx_usada, n_ctx);
      switch (estrategia) 
      {
        case COMPLETO:
            //llama_kv_cache_clear(gestor->ctx);
            printf("************ Estrategia Limpieza completa *******\n");
            llama_memory_seq_rm(llama_get_memory(contexto), 0, n_sys_tokens, -1);  // Limpia conservando Rol-System
            *n_past = llama_memory_seq_pos_max(llama_get_memory(contexto), 0) + 1;
            break;

        case VENTANA_DESLIZANTE:
           printf("******* Estrategia Ventana Deslizante... %d *******\n",n_tokens_a_borrar);
           // Técnica avanzada de "Sliding Window" (Ventana Deslizable).
           // Límites de la "poda" p0(Despues del ROL) p1 + borrado
           int32_t p0 = n_sys_tokens;   //Fin ROL Desde donde borraremos
           int32_t p1 = p0 + n_tokens_a_borrar;  /// Hasta donde borramos avanzando los tokens a borrar
           llama_memory_seq_rm(llama_get_memory(contexto), 0, p0, p1); // Elimina bloque antiguo
    
           // Movemos lo que quedó al final hacia atrás para cerrar el hueco asi tendremos la memoria contigua
           llama_memory_seq_add(llama_get_memory(contexto), 0, p1, -1, -n_tokens_a_borrar);
    
           // CRITICO: Recalcular n_past para que coincida con la nueva realidad del KV Cache
           // Tras el desplazamiento, la nueva máxima posición es la antigua menos el desplazamiento.
           *n_past = llama_memory_seq_pos_max(llama_get_memory(contexto), 0) + 1;
    
           printf("Ventana deslizante aplicada con éxito. Nueva posición n_past: %d\n", *n_past);
           break;
      }
   }
}


////////////////////////////////////////////////  FUNCIONES publica //////////////////////////////////
/***************************** Inicia el gestor indicando si queremos o no LOG ****************/
void  iniciaGestor(bool log)
{
   printf("Inicializamos gestor con log %d\n",log);
   if(!log){ llama_log_set(mi_funcion_log_silenciosa, NULL);}
   llama_backend_init();  // *********** Iniciamos Backend

   // Inicializamos lista de modelos que se guarda en la estatica gestor.
   GestorModelos lista;
   lista.cabeza = NULL;
   lista.modelos = 0;
   gestor = lista;
}

/***************************** Limpia los datos del Gestor asi como backend ****************/
void  destructorGestor(void)
{
   limpiaGestor();
   llama_backend_free();
}

/*********************************************************************************************
 * Realiza la carga del modelo desde fichero y lo guarda en la lista del gestor
 *
 * Parametros: El nombre del modelo y el camino al fichero .GUUF
 *
 * Retorno:  positivo es el identificador del modelo con el que trabajaremos.
 *          -1  Alcanzo limite de modelos
 *          -2 Error al cargar el fichero del modelo.
 *          -3 Memoria insuficiente para la carga del modelo
 *          -4 Error al asignar el contexto
 ********************************************************************************************/
int cargaModelo(const char *nombre, char* pathModelo)
{
   if (gestor.modelos >= MAX_MODELOS) { fprintf(stderr, "Error: Se alcanzó el límite de %d modelos.\n", MAX_MODELOS); return -1; }

   // ** 1.- **  Carga Modelo
   // Primero definimos los parametros
   paramModelo_t parametrosModelo = llama_model_default_params();
   parametrosModelo.n_gpu_layers = 0;  // Fuerza CPU
   parametrosModelo.progress_callback = mi_progreso; // Proceso carga

   modelo_t *modelo = llama_model_load_from_file(pathModelo, parametrosModelo);  // carga del modelo en memoria.
   if (modelo == NULL) { fprintf(stderr, "Error: No se pudo cargar el modelo '%s'.\n", nombre); return -2; }

   // ** A ** Registrar el modelo en el gestor
   Nodo *nuevo = (Nodo *) malloc(sizeof(Nodo)); // Reservar memoria para el nuevo nodo. 
   if (nuevo == NULL) { fprintf(stderr, "Error: memoria insuficiente\n"); return -3; }

   // ** B ** Rellenar sus campos. 
   nuevo->dato.id = gestor.count;  // El identificador numero incremental
   strncpy(nuevo->dato.nombre,nombre, sizeof(nuevo->dato.nombre) -1);
   strncpy(nuevo->dato.pathModelo,pathModelo, sizeof(nuevo->dato.pathModelo) -1);
   nuevo->dato.modelo = modelo;
   nuevo->dato.contexto = NULL;

   // ** 2.- ** Contexto por defecto (podra cambiarse con get_ParamContexto/put_ParamContexto)
   //paramContext_t parametrosContexto = llama_context_default_params();
   nuevo->dato.params_ctx = llama_context_default_params();     // Carga estructura con paramentros por defecto
   // Optimización para OpenBLAS en CPU
   nuevo->dato.params_ctx.n_ctx = 4096;   
   nuevo->dato.params_ctx.n_batch = 4096; // Tokens a procesar para llama_decode. "lote para el prompt" multiplo de n_ubatch
   nuevo->dato.params_ctx.n_ubatch = 512; // Punto dulce en CPU no muy alto

   nuevo->dato.params_ctx.n_threads = 4;  // Hilos para generación en CPU los fisicos, GPU no poner
   nuevo->dato.params_ctx.n_threads_batch = 4;       // Hilos para lotes
   nuevo->dato.params_ctx.offload_kqv = false;      // true en GPU
   nuevo->dato.params_ctx.embeddings = false;        // No extraer embeddings
   nuevo->dato.params_ctx.rope_freq_base = 0.0f;     // Usar valor del modelo
   nuevo->dato.params_ctx.rope_freq_scale = 0.0f;    // Usar valor del modelo
   context_t *contexto = llama_init_from_model(nuevo->dato.modelo, nuevo->dato.params_ctx);
   if (contexto == NULL) {return -4; } 
   nuevo->dato.contexto = contexto;
   
   nuevo->siguiente = NULL;

   printf("Modelo '%s' cargado exitosamente \n", nombre);

   // ** C ** Asigna nodo. Si la lista está vacía, el nuevo nodo es la cabeza, sino al final. 
   if (gestor.cabeza == NULL) { gestor.cabeza = nuevo; } 
   else 
   {
      // Si no, recorrer hasta el último nodo y enlazar.
      Nodo *actual = gestor.cabeza;
      while (actual->siguiente != NULL) { actual = actual->siguiente; }
      actual->siguiente = nuevo;
    }
   gestor.modelos++;

   return gestor.count++;
}

/*********************************************************************************************
 * Elimina Modelos de la lista, pasado ID
 *
 * Parametro identificador del modelo.
 *
 * Retorno  Da el numero de modelos que quedn en memoria
 *         -1 el gestor esta vacio
 *         -2 No se encontro modelo con identificador pasado
 ********************************************************************************************/
int borraModelo(int id)
{
   if (gestor.cabeza == NULL) { return -1; }   // Gestor sin modelos  
   Nodo *actual = gestor.cabeza;
   Nodo *anterior = NULL;

   // Buscar el nodo 
   while (actual != NULL && actual->dato.id != id) 
   {
       anterior = actual;
        actual = actual->siguiente;
   }

   if (actual == NULL) { return -2; }  // no se encontró 

   /* Reenlazar: saltarse el nodo a borrar */
   if (anterior == NULL) {  gestor.cabeza = actual->siguiente; } // Era el cabecera
   else { anterior->siguiente = actual->siguiente; }

   /* Liberar la memoria del nodo */
   limpiaModelo(&actual->dato);
   free(actual);   
   gestor.modelos--;

   return gestor.modelos;
}


/*********************************************************************************************
 * Lista los modelos cargados
 ********************************************************************************************/
void listaModelos(void)
{
    Nodo *actual = gestor.cabeza;

   printf("================== %d  MODELOS CARGADOS =================================\n",gestor.modelos);
   while (actual != NULL) 
   {
      printf("ID=%d \tNombre: %s \t\t Archivo: %s\n", actual->dato.id, actual->dato.nombre, 
                                                      actual->dato.pathModelo); 
      actual = actual->siguiente;
   }
   printf("======================================================================\n");
}


/*********************************************************************************************
 * Deja el gestor vacio liberando todos los modelos cargados
 ********************************************************************************************/
void limpiaGestor(void) 
{
   Nodo *actual = gestor.cabeza;
   while (actual != NULL) 
   {
       Nodo *siguiente = actual->siguiente;
       limpiaModelo(&actual->dato);
       free(actual);
       actual = siguiente;
   }
   gestor.cabeza = NULL;
   gestor.modelos = 0;
}

/****************  Nos da los modelos cargador  **********************/
int modelosCargados(void) { return gestor.modelos; }

/*********************************************************************************************
 * Nos de el nombre del modelo con identificador pasado por parametro
 *
 * Parametro id identificador del modelo
 *
 * retorno nombre del modelo o "" si no se encuentra.
 ********************************************************************************************/
char *get_NombreModelo(int id) 
{ 
    Nodo *nodo = buscaNodo(id);
    if (nodo == NULL) { return ""; }
    
    return nodo->dato.nombre; 
}

/*********************************************************************************************
 * Nos de el nombre del archivo del modelo
 *
 * Parametro id identificador del modelo
 *
 * retorno nombre del modelo o "" si no se encuentra.
 ********************************************************************************************/
char *get_NombreArchivo(int id) 
{ 
    Nodo *nodo = buscaNodo(id);
    if (nodo == NULL) { return ""; }
    
    return nodo->dato.pathModelo; 
}

/*********************************************************************************************
 *     Da el numero total de vocabulario TOKENs del modelo
 ********************************************************************************************/
int32_t vocab_n_tokens(int id) 
{
    Nodo *nodo = buscaNodo(id);
    const struct llama_vocab *vocab = llama_model_get_vocab(nodo->dato.modelo);
    return llama_vocab_n_tokens(vocab);
}


/*********************************************************************************************
 * Nos da los parametros de contexto que estan configurados
 ********************************************************************************************/
paramContext_t get_ParamContexto(int id) 
{ 
    Nodo *nodo = buscaNodo(id);
    if (nodo == NULL) { paramContext_t porDefecto = llama_context_default_params(); return porDefecto;}
    
    return nodo->dato.params_ctx; 
}
/*********************************************************************************************
 * Cambiamos los parametros de contexto, antes limpiaremos el contexto
 ********************************************************************************************/
int put_ParamContexto(int id, paramContext_t parametros) 
{
    Nodo *nodo = buscaNodo(id);
    if (nodo == NULL) { return -1;}
   // Lo primero liberamos el contexto
   llama_free(nodo->dato.contexto);

   // Ahora se cambia
   context_t *contexto = llama_init_from_model(nodo->dato.modelo, parametros);
   if (contexto == NULL) { return -2; } 
   nodo->dato.contexto = contexto;
   nodo->dato.params_ctx = parametros;

   return 0;
}

/*********************************************************************************************
 * Realiza la inferencia a una pregunta
 *
 * Parametros:
 *   id: 		Identificador del modelo
 *   pregunta:		Pregunta que se realiza
 *   rolSystem:		Rol del sistema, marca ROL
 *   sampler:		Cadena de samplers
 *   respuesta:		Resultado de la inferencia si OK un String dinamico
 *   limpio:		Si es true limpiamos el contexto de tokens
 *   limTokens:		Limete de tokens
 *   estrategia:	Estrategia de limpieza del contexto.
 *
 * retorno Negativo error: -1 ID erroneo, -2 error al tokenizar, -3, process inicial, 
 *                         -4 precess del bucle, -5 al destokenizar, -6 No pudo redimensionar respuesta
 ********************************************************************************************/
int inferir(int id,  const char *pregunta, char *rolSystem, sampler_t *sampler, StringDinamico_t *respuesta, 
            bool limpio, int limTokens, LimpiaContexto_t estrategia)
{
    //*** Lo primero vemos si se aumenta respuesta, y se pone en cadena vacia
    size_t tam_inicial = (size_t)(limTokens * 1) ; // Tamño estimado segun TOKENs
    if (respuesta->dato == NULL || respuesta->capacidad < tam_inicial)  { ampliaString(respuesta, tam_inicial); }
    //  Iniciamos el array a cadena vacia
    respuesta->dato[0] = '\0';
    respuesta->longitud = 0;

    //**** Localizamos Modelo, obteniendo modelo y contexto que seran usados en la inferencia
    Nodo *nodo = buscaNodo(id);
    if(nodo == NULL) { return -1; }
    ModeloInfo modeloInfo = nodo->dato;
    modelo_t *modelo = modeloInfo.modelo;
    context_t *contexto = modeloInfo.contexto;
 
    //**** Si se marca como limpio, entonces reiniciamos contexto.
    if(limpio) { llama_memory_clear(llama_get_memory(contexto), true); }
 
    // ** 1 ** Cargamos el vocabulario del modelo
    vocab_t *vocabulario = (vocab_t *) llama_model_get_vocab(modelo);
 
    // ** 2 ** Tokenizamos RolSystem + User
    // Procesamos el prompt
    mensajeChat_t rol_System[] = {{"system", rolSystem }};    // define el rol de como comportarse.
    mensajeChat_t rol_User[] = {{"user",   pregunta}};          // La pregunta
    int32_t n_sys_tokens = calculaTokensROL(rol_User, rol_System, modelo, vocabulario);
 
    // **** Si es el primer mensaje se añade tanto el rol_system como el rol_user, en los siguientes solo User
    bool elPrimero = llama_memory_seq_pos_max(llama_get_memory(contexto), 0) == -1;
    int n_mensajes ;
    mensajeChat_t mensajes[2];  // Preparado para ROL y User
    if(elPrimero)
    {
       mensajes[0] = rol_System[0];
       mensajes[1] = rol_User[0];
       n_mensajes = 2; // Tenemos dos mensajes en el historial
    }
    else
    {
       mensajes[0] = rol_User[0];
       n_mensajes = 1; // Tenemos 1
    }
 
    // **********  Generamos el Prompt Formateado con el mensaje
    const char * tmpl = llama_model_chat_template(modelo, NULL);  // Plantilla
    int32_t bytes_prompt_final = llama_chat_apply_template(tmpl, mensajes, n_mensajes, true, NULL, 0);  // calculo del buffer necesario
    char *prompt_formateado = malloc(bytes_prompt_final + 1);
    llama_chat_apply_template(tmpl, mensajes, n_mensajes, true, prompt_formateado, bytes_prompt_final);  // Aplica
    prompt_formateado[bytes_prompt_final] = '\0'; // Asegurar cierre C
 
    // **********  Tokenizamos el prompt formateado con mensaje
    // Cuando pedimos sin buffer de salida por logica no tiene espacio, así que dara un negativo.
    int32_t tokens_necesarios_negativos = llama_tokenize(vocabulario,prompt_formateado,strlen(prompt_formateado),NULL,0,elPrimero,true);
    int32_t n_prompt = tokens_necesarios_negativos * -1;
    llama_token *prompt_tokens = (llama_token*) malloc(n_prompt*sizeof(llama_token));
    if (llama_tokenize(vocabulario,prompt_formateado,strlen(prompt_formateado),prompt_tokens,n_prompt,elPrimero,true)<0 ) { fprintf(stderr,"%s: Fallo al tokenizar\n",__func__); return -2 ; }
    free(prompt_formateado); // Ya no es neceserio por lo tanto lo borramos cuanto antes.
 
     // ** 3 ** Batch de la inferencia inicial
    int32_t start_pos = 0;
    if (!elPrimero) 
    {
        llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(contexto), 0);
        start_pos = (max_pos == -1) ? 0 : max_pos + 1;
    }
    // Creamos el batch1 manualmente para controlar las posiciones 'pos'
    // 3.1. Inicializamos el batch extendido vinculado al contexto
    batch_ext_t *miBatch1 = llama_batch_ext_init(contexto);
    int n_tokens = n_prompt;
    llama_batch_ext_clear(miBatch1); // Limpiamos por seguridad
 
    // 3.2. Llenamos el batch token por token
    for (int i = 0; i < n_prompt; i++) 
    {
       // Añadimos el token a la secuencia 0. 'idx' es la posición interna asignada.
       int32_t idx = llama_batch_ext_add_token(miBatch1, 0, prompt_tokens[i]);
     
       // Establecemos la posición (pos) para la continuidad del caché KV
       llama_pos pos = start_pos + i;
       llama_batch_ext_set_pos(miBatch1, idx, &pos);
    }
    // 3.3. Activamos los logits SOLO para el último token
    // El último token añadido está en el índice (n_tokens - 1)
    llama_batch_ext_set_output_logits(miBatch1, n_prompt - 1, true);
 
 
    // 3.4. Ejecutamos la inferencia (reemplazo directo de llama_decode)
    // En nuevo API definimos el tipo de proceso nosotros texto LLAMA_PROCESS_TYPE_DECODE
    if (llama_process(contexto, LLAMA_PROCESS_TYPE_DECODE, miBatch1) != 0) 
    {
       fprintf(stderr, "Error crítico: falló el proceso del prompt inicial.\n");
       llama_batch_ext_free(miBatch1);
       free(prompt_tokens);
       return -3;
    }
 
    // 3.5. Liberamos la memoria del batch y de los tokens
    llama_batch_ext_free(miBatch1);
    free(prompt_tokens);
 
 
    // ** 4 ** Bucle inferencia
    //Variables de control del bucle
    int n_predict = MAXTOKENS;       // Máximo de tokens a generar
    if (limTokens != 0) { n_predict = limTokens ;}
    llama_token ID_nuevoToken;
    int n_generados = 0;          // Contador de tokens generados
    int n_past = start_pos + n_tokens;        // Posición absoluta actual en la secuencia (KV Cache, contexto)
 
    // ==========================================================
    // 4.1. FUERA DEL BUCLE: Inicializar el batch UNA SOLA VEZ
    // ==========================================================
    // La API gestiona internamente la memoria. No necesitamos asignar manualmente
    // n_tokens, n_seq_id, seq_id o logits aquí.
    batch_ext_t *miBatch_Sig = llama_batch_ext_init(contexto);
    n_tokens = 1;
 
    // ==========================================================
    // 4.2. BUCLE DE GENERACIÓN AUTORREGRESIVO
    // ==========================================================
    for (; n_generados < n_predict; ++n_generados)
    {
        // A. Muestreo del siguiente token (usa los logits del último token evaluado)
        ID_nuevoToken = llama_sampler_sample(sampler, contexto, -1);
    
        // B. Condición de parada: Fin de Generación (EOS / EOT)
        if (llama_vocab_is_eog(vocabulario, ID_nuevoToken)) { break; }
    
        // C. Destokenización: convertimos TOKENs a Texto
        char buf[n_predict];
        int n = llama_token_to_piece(vocabulario, ID_nuevoToken, buf, sizeof(buf), 0, true);
        
        if (n < 0) 
        { 
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__); 
            respuesta->dato[0] = '\0';
            // ¡Crítico: Liberar memoria antes de salir abruptamente!
            llama_batch_ext_free(miBatch_Sig); 
            return -5; 
        }
        buf[n] = '\0';
        
        // Miramos si debemos redimensionar el StringDinamico
        size_t espacio_necesario = n + 1; // +1 para el '\0'
        size_t espacio_disponible = respuesta->capacidad - respuesta->longitud;
        if (espacio_necesario > espacio_disponible) 
        {
            // Calcular cuántos chunks necesitamos
            size_t deficit = espacio_necesario - espacio_disponible;
            size_t chunks_adicionales = (deficit + CHUNK_SIZE - 1) / CHUNK_SIZE;
            size_t expansion = chunks_adicionales * CHUNK_SIZE;
            
            ampliaString(respuesta, expansion);
        }

        // Copiamos fragmento
        cadenaAString(respuesta,buf);
         
        // D. Preparar el batch para el SIGUIENTE token generado y lo decodifica
        // 1. Limpiar el batch de la iteración anterior (resetear contadores, no libera memoria)
        llama_batch_ext_clear(miBatch_Sig);
        
        // 2. Añadir el nuevo token a la secuencia 0. 'idx' será 0.
        int32_t idx = llama_batch_ext_add_token(miBatch_Sig, 0, ID_nuevoToken);
        
        // 3. Establecer la posición actual (n_past)
        llama_pos pos = n_past;
        llama_batch_ext_set_pos(miBatch_Sig, idx, &pos);
        
        // 4. Activar logits para este token (equivalente exacto a tu antiguo: miBatch_Sig.logits[0] = true)
        llama_batch_ext_set_output_logits(miBatch_Sig, idx, true);
        
        // 5. Ejecutar la inferencia (reemplazo directo de llama_decode)
        if (llama_process(contexto, LLAMA_PROCESS_TYPE_DECODE, miBatch_Sig) != 0) 
        { 
            fprintf(stderr, "Error crítico: falló el proceso del token generado.\n"); 
            llama_batch_ext_free(miBatch_Sig); 
            return -4 ; 
        }
    
        // E. Actualizar la posición absoluta para la próxima iteración
        n_past++;
       
        // F.   Analizamos el espacio del contexto
        gestContexto(contexto, miBatch_Sig, 1, &n_past, n_sys_tokens,256,rolSystem, estrategia) ;
    
    }  // Fin del bucle de inferencia
 
    respuesta->dato[respuesta->longitud] = '\0';
    llama_batch_ext_free(miBatch_Sig);
 
    return 0;  // Si llega al final es OK
}

