/*********************************************************************
 * Tipos a utilizar en el desarrollo de llama
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 **********************************************************************/
#ifndef TIPOSLLAMACPP_H
#include "llama.h"
#define TIPOSLLAMACPP_H

    // Parametros de configuración
    typedef struct llama_model_params paramModelo_t;
    typedef struct llama_context_params paramContext_t;
    typedef struct llama_sampler_chain_params paramSampler_t;

    // Estructuras principales
    typedef struct llama_model modelo_t;
    typedef struct llama_context context_t;
    typedef struct llama_vocab vocab_t;
    typedef struct llama_sampler sampler_t;
    typedef struct llama_batch batch_t;

    // Mensajes y Tokens
    typedef struct llama_chat_message mensajeChat_t;



#endif
