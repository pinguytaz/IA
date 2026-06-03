/***********************************************************************************
 * Base		Programa para explicación de la programación de llama.cpp
 *              en C, con funciones llama_, esto supone que no usaremos intrucciones 
 *		de bajo nivel "Ggml" que es una biblioteca de tensores.
 *		"make" preparado para compilar en Linux o con compilación cruzada
 *		a Windows.
 *		Para Intel podremos usar oneAPI(SYCL) y openBLAS.
 *		Llama.cpp es un motor de inferencia para LLM, de código abierto.
 *		
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 *	Tendremos que descargarnos un modelo preentrenado en formato GGUF, por ejemplo
 *	de HuggingFace .gguf, o podremos convertirlo.
 *	Buscar en HuggingFace <modelo>+ggufa cuantificación Q4_K_M es muy apto.
 *	Modelos: Llama, Mistral, Qwen, Whisper, Gemma, DeepSeek,  etc.
 *
 *	Tambien podremos bajarnos cualquier otro y cuantizarlo con las herramientas
 *	de llama.cpp
 *
 * librerias:
 *	llama.cpp	Libreria que nos permite ganerar inferencia LLM en C
 ***********************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// Libreria llama.cpp
#include "llama.h"

// Nuestros ficheros auxiliares para definir tipos y funciones para el proyecto.
#include "../includes/tiposLlamacpp.h"
#include "funcPruebas.h"
#include "../includes/funcCallB.h"

int main(int argc, char ** argv) 
{
    if (argc != 3)
    {
        printf("Uso: %s modelo rol prompt\n", argv[0]);
	return -1;
    }

    const char *pathModelo = argv[1];
    const char *prompt = argv[2];

    printf("Inicializando sistema IA...\n");
    llama_log_set(mi_funcion_log_silenciosa, NULL);  // Silencioamos LOG

    /* 1.- *********** Inicializar el backend de Llama ****************************/
    printf("1.- Cargamos Backends \n");
    llama_backend_init();
    printf("\t¡Backends inicializados correctamente!\n");

    /* 2.- ********** Cargar el modelo en memoria *********************************/
    printf("2.- Cargamos Modelo a memoria %s\n",pathModelo);

    /* 2.1- **********  Configuración de prámetros. ********************************/
    printf("\t2.1 Configuramos parametros del Modelo \n");
    // Inicializamos y cargamos el modelo pasado por parametro
    paramModelo_t parametrosModelo = llama_model_default_params();
    //  Forzar ejecución en CPU 0 capas a laCPU se procesa en CPU
    parametrosModelo.n_gpu_layers = 0; 
    //  Optimización de memoria para CPU 
    // Permite compartir la memoria del modelo eficientemente
    parametrosModelo.use_mmap  = true; 
    // EVITA que el sistema operativo mande el modelo al disco (Swap)
    // ¡Ojo! suficiente RAM libre para el modelo. sino a false
    parametrosModelo.use_mlock = false; 
			       
    //Ponemos funcion de progeso de carga.
    parametrosModelo.progress_callback = mi_progreso; 
    printf("\t!Parametros establecidos \n");
    
    /* 2.2- ********** Cargar el modelo en memoria *********************************/
    modelo_t *modelo = llama_model_load_from_file(pathModelo, parametrosModelo);
    if (modelo == NULL) 
    {
        fprintf(stderr, "Error al cargar el modelo.\n");
        return -1;
    }
    
    infoModelo(modelo);
    printf("\t¡Modelo cargado y configurado!\n");

    /* 3.- ********** Configuracion e inicialización del contexto. ****************/
    printf("3.- Configuramos contexto \n");
    paramContext_t parametrosContexto = llama_context_default_params();
    // Optimización para OpenBLAS en CPU
    parametrosContexto.n_ctx = 2048;  // Tamaño contexto
    //parametrosContexto.n_batch = 2048; // Tokes a procesar para llama_decodee.
    parametrosContexto.n_threads = 4; // Nucleos fisicos CPU 
    parametrosContexto.n_threads_batch = 4; // Hilos processo 
    parametrosContexto.flash_attn_type = true; //Ahorra RAM de contexto 
    // .no_pref = habilta contadores tiempos

    // Crear el contexto ligado al modelo
    context_t *contexto = llama_init_from_model(modelo, parametrosContexto);

    if (contexto == NULL) 
    {
        fprintf(stderr, "ERROR: No se pudo crear el contexto de llama.\n");
        return 1;
    }
    printf("\t¡Contexto de la sesión creado con éxito!\n");
 
    /* 4.- ********* TOKENIZADOR **************************************************/
    printf("4.- Preparamos tokenizador\n");
    // Lo primero sera cargar el vocabulario.
    const vocab_t *vocab = llama_model_get_vocab(modelo);
    int32_t tamaVocabulario = llama_vocab_n_tokens(vocab);
    printf("\tCarga vocabulario del modelo con un total de %d tokens únicos.\n", tamaVocabulario);


    // Preparamos prompt con su rol de system y user
    const char *rolSystem = "La forma de comunicarme es en Español";
    mensajeChat_t mensajes[] = {
          {"system", rolSystem },
          {"user",   prompt}
      };

    int n_mensajes = 2; // Tenemos dos mensajes en el historial
    int32_t bytes_prompt_final = llama_chat_apply_template(llama_model_chat_template(modelo, NULL), mensajes, n_mensajes, true, NULL, 0);
    char * prompt_formateado = malloc(bytes_prompt_final + 1);
    llama_chat_apply_template(llama_model_chat_template(modelo, NULL), mensajes, n_mensajes, true, prompt_formateado, bytes_prompt_final);
    prompt_formateado[bytes_prompt_final] = '\0'; // Asegurar cierre de cadena de C
		
    // Continuamos con el Tokenizador
    int32_t tokens_necesarios_negativos = llama_tokenize(vocab,prompt_formateado,strlen(prompt_formateado),NULL,0,false,true);
    int32_t n_tokens = tokens_necesarios_negativos *  -1;

    printf("\tEl PROMPT tiene: %d TOKENS\n",n_tokens);

    llama_token *prompt_tokens = (llama_token*) malloc(n_tokens*sizeof(llama_token));
    if (llama_tokenize(vocab,prompt_formateado,strlen(prompt_formateado),prompt_tokens,n_tokens,false,true)<0 )
    {
        fprintf(stderr,"%s: Fallo al tokenizar\n",__func__);
	return -2;
    }

    //desgloseTOKENS(vocab,n_tokens, prompt_tokens);

    printf("\t¡Tokenizado preparado con su vocabulario!\n");

    /* 5.- ******* Crea Sampler **************************************/
    printf("5.- Preparamos sampler\n");
    // Iniciamos Sampler
    paramSampler_t sparams = llama_sampler_chain_default_params(); 
    sparams.no_perf = false;
    sampler_t *miSampler = llama_sampler_chain_init(sparams);
    //llama_sampler_chain_add(miSampler, llama_sampler_init_greedy()); // Elige la mas exata

    llama_sampler_chain_add(miSampler, llama_sampler_init_min_p(0.05f, 1)); // Min-P
    llama_sampler_chain_add(miSampler, llama_sampler_init_temp(0.8f));   //Temperatura
    //llama_sampler_chain_add(miSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED)); // Distribución
 
    printf("\t¡Sampler  preparado!\n");


    /* 6- ******* Crea BATCH ********************************************/
    printf("6- preparamos batch\n");
    llama_batch miBatch = llama_batch_get_one(prompt_tokens, n_tokens);

    infoBatch(miBatch);

    //  ¿El modelo tiene un "Encoder" (Codificador)?
    // Los modelos puros de texto como Llama o Mistral devuelven 'false' aquí.
    // Los modelos como Whisper (audio) o modelos de traducción devuelven 'true'.
    if (llama_model_has_encoder(modelo)) 
    {
        printf("\tModelo audio o traducción\n");
        // Si tiene encoder, procesamos el prompt 
        if (llama_encode(contexto, miBatch)) 
	{
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        // Buscamos cuál es el token oficial que arranca su fase de generación
        llama_token decoder_start_token_id = llama_model_decoder_start_token(modelo);

        // Si el modelo no define un token de inicio específico 
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) 
	{
            // usamos por defecto el clásico token BOS (Beginning of Sequence)
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        // Reutilizamos la lógica creando un nuevo batch UNITARIO (de 1 solo token)
        // que contiene únicamente el token de arranque 
        miBatch = llama_batch_get_one(&decoder_start_token_id, 1);
    }
    else
    {
        printf("\tModelo de texto ¿llama?\n");
    }
    printf("\t¡Batch  preparado!\n");


    /* 7- ******* Crea BATCH ********************************************/
    printf("7.- Bucle inferencia\n");
    int n_predict = 256;
    llama_token new_token_id;
    for (int n_pos = 0; n_pos + miBatch.n_tokens < n_tokens + n_predict; ) 
    {
        if (llama_decode(contexto, miBatch) != 0) 
        {
            fprintf(stderr, "Error crítico: falló el decode del prompt inicial.\n");
            return 1;
        }
	n_pos += miBatch.n_tokens;
        
        new_token_id = llama_sampler_sample(miSampler, contexto, -1);
        // Esto es el fin de la generación
        if (llama_vocab_is_eog(vocab, new_token_id)) 
        {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) 
        {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        buf[n] = 0;
        printf("%s", buf);

        // Prepara el sigiente lote con TOKEN sample
        miBatch = llama_batch_get_one(&new_token_id, 1);
    }

    printf("\n\t¡Fin bucle inferencia!\n");

    // 9.- ***************** Limpieza al terminar
    printf("\n--- 9. Limpia memoria ---\n");
    llama_sampler_free(miSampler);
    free(prompt_tokens);
    free(prompt_formateado); 
    llama_free(contexto);
    llama_model_free(modelo);
    llama_backend_free();

    printf("***************** ¡Fin programa IA! ******************\n");
    return 0;
}
