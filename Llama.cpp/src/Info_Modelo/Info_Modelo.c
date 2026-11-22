/***********************************************************************************
 * Base		Programa Nos da la información de un modelo, para ayudarnos a configurar
 *              los parametros en nuestros programas
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

void mi_funcion_log_silenciosa(enum ggml_log_level level, const char * text, void * user_data);   // Captura log y no hace nada
void daEtiquetaMetaModelo(modelo_t *modelo, const char *clave, char *buf, char *arquitectura);

int main(int argc, char ** argv) 
{
    if (argc < 2)
    {
        printf("Uso: %s modelo \n", argv[0]);
	return -1;
    }

    const char *pathModelo = argv[1];

    printf("Inicializando sistema IA...\n");
    llama_log_set(mi_funcion_log_silenciosa, NULL);  // Silenciamos LOG

    // Cargamos el Backend
    llama_backend_init();

    // Cargamos el modelo
    printf("Cargamos Modelo %s \n",pathModelo);

    // Parametros
    paramModelo_t parametrosModelo = llama_model_default_params();
    parametrosModelo.n_gpu_layers = 0; //  Forzar ejecución en CPU 0 capas
    parametrosModelo.vocab_only = true; //  Solo vocabulario, ya que no lo ejecutaremos solo queremos información del modelo.

    // Carga del modelo
    modelo_t *modelo = llama_model_load_from_file(pathModelo, parametrosModelo);
    if (modelo == NULL) { fprintf(stderr, "Error al cargar el modelo.\n"); }


    printf("\n============= INFORMACIÓN MODELO GGUF =============\n");
    printf("Modelo analizado: %s\n", pathModelo);
    char buffer_meta[512];
    char arquitectura[80];

    printf("Modelo analizado: %s\n", pathModelo);
    daEtiquetaMetaModelo(modelo, "general.name", buffer_meta,NULL); printf("\tNombre: %s", buffer_meta);
    daEtiquetaMetaModelo(modelo, "general.basename", buffer_meta,NULL); printf("\tFamilia: %s", buffer_meta);
    daEtiquetaMetaModelo(modelo, "general.license", buffer_meta,NULL); printf("\tLicencia: %s", buffer_meta);
    daEtiquetaMetaModelo(modelo, "general.architecture", arquitectura,NULL); 
    if(strcmp(arquitectura,"mistral") == 0){strcpy(arquitectura,"mistral3"); }
    printf("\tArquitectura base: %s\n", arquitectura);
    daEtiquetaMetaModelo(modelo, "general.type", buffer_meta,NULL); printf("\tTipo de modelo: %s\n", buffer_meta);
    daEtiquetaMetaModelo(modelo, "general.finetune", buffer_meta,NULL); printf("\tTipo fine-tuning: %s\n", buffer_meta);

    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║        ARQUITECTURA DEL MODELO CARGADO        ║\n");
    printf("╠═══════════════════════════════════════════════╣\n");
    daEtiquetaMetaModelo(modelo, "general.size_label", buffer_meta,NULL); 
    printf("║ Parametros\t: %s (%lu)\t\t║\n", buffer_meta, (unsigned long) llama_model_n_params(modelo)); 
    //  Tamaño del modelo
    uint64_t bytes_modelo = llama_model_size(modelo);
    double gb_modelo = ((double)bytes_modelo / (1024.0 * 1024.0)) / 1024.0;
    printf("║ Tamaño del modelo: %lu MB /%.0f Gigas\t\t║\n", bytes_modelo/(1024*1024), gb_modelo); 

    daEtiquetaMetaModelo(modelo, "block_count", buffer_meta,arquitectura); 
    printf("║ Capas transformer (max n_gpu_layers): %s\t║\n", buffer_meta); 

    daEtiquetaMetaModelo(modelo, "context_length", buffer_meta,arquitectura); 
    printf("║ Maxima de contexto(n_ctx): %s tokens\t║\n", buffer_meta); 
    //int32_t max=llama_n_ctx_train(modelo);
    //printf("------------- %d\n",max);

    daEtiquetaMetaModelo(modelo, "embedding_length", buffer_meta,arquitectura); 
    printf("║ Embedding (uso de memoria): %s \t\t║\n", buffer_meta); 
    daEtiquetaMetaModelo(modelo, "attention.key_length", buffer_meta,arquitectura); 
    printf("║ Dimensión Clave: %s", buffer_meta); 
    daEtiquetaMetaModelo(modelo, "attention.value_length", buffer_meta,arquitectura); 
    printf(" Valor: %s\t\t║\n", buffer_meta); 
    daEtiquetaMetaModelo(modelo, "general.file_type", buffer_meta,NULL); 
    printf("║ Tipo de cuantización: %s\t\t\t║\n", buffer_meta); 
    daEtiquetaMetaModelo(modelo, "rope.freq_base", buffer_meta,arquitectura); 
    printf("║ RoPE frecuencia base: %s\t\t\t║\n", buffer_meta); 

    printf("╠═══════════════════════════════════════════════╣\n");
    printf("║            Información tokenizador            ║\n");
    printf("╠═══════════════════════════════════════════════╣\n");
    const struct llama_vocab * vocabulario = llama_model_get_vocab(modelo);
    enum llama_vocab_type tipo = llama_vocab_type(vocabulario);
    const char * nombre_tipo;
    switch (tipo) 
    {
       case LLAMA_VOCAB_TYPE_NONE: nombre_tipo = "NONE (sin vocabulario)"; break;
       case LLAMA_VOCAB_TYPE_SPM:  nombre_tipo = "SPM (SentencePiece) (LlaMA)"; break;
       case LLAMA_VOCAB_TYPE_BPE:  nombre_tipo = "BPE (Byte Pair Encoding) (GPT-2)"; break;
       case LLAMA_VOCAB_TYPE_WPM:  nombre_tipo = "WPM (WordPiece) (BERT)"; break;
       case LLAMA_VOCAB_TYPE_UGM:  nombre_tipo = "UGM (Unigram)"; break;
       case LLAMA_VOCAB_TYPE_RWKV: nombre_tipo = "RWKV"; break;
       case LLAMA_VOCAB_TYPE_PLAMO2: nombre_tipo = "PLaMo-2"; break;
       case LLAMA_VOCAB_TYPE_TEST: nombre_tipo = "TEST (dummy)"; break;
       default: nombre_tipo = "Desconocido";
   }
   printf("║ Tipo: %s\t║\n",nombre_tipo);
   printf("║ Tokens vocabulario: %d\t\t\t║\n",llama_vocab_n_tokens(vocabulario));
   // === TOKENES ESPECIALES FUNDAMENTALES ===
   llama_token token_id;

   // TOKEN BOS Inicio secuencia
   token_id =llama_vocab_bos(vocabulario);  // Inicio Secuencia
   printf("║ BOS (inicio) %6d\t\t\t\t║\n",token_id); 
   printf("║\t%s\t\t\t║\n",llama_vocab_get_text(vocabulario, token_id)); 
   // EOS Fin
   token_id =llama_vocab_eos(vocabulario);  // Inicio Secuencia
   printf("║ EOS (fin) %6d\t\t\t\t║\n", token_id);  // Fin secuencia
   printf("║\t%s\t\t\t\t║\n",llama_vocab_get_text(vocabulario, token_id)); 
   // EOT Fin de turno 
   token_id =llama_vocab_eot(vocabulario);  // Inicio Secuencia
   printf("║ EOT (fin turno) %6d\t\t\t║\n", token_id);  // End Of Turn (conversacional)
   printf("║\t%s\t\t\t\t║\n",llama_vocab_get_text(vocabulario, token_id)); 
   printf("╚═══════════════════════════════════════════════╝\n");

    // Liberamos recursos
    llama_model_free(modelo);
    llama_backend_free();
    return 0;
}


void mi_funcion_log_silenciosa(enum ggml_log_level level, const char * text, void * user_data) {  ;  }
void daEtiquetaMetaModelo(modelo_t *modelo, const char *clave, char *buf,char *arquitectura)
{
    char metaDato[128];   // Tendra nombre de metadato, para cuendo depende de arquitectura.
    if (arquitectura != NULL) { snprintf(metaDato,sizeof(metaDato), "%s.%s", arquitectura,clave); }
    else {snprintf(metaDato,sizeof(metaDato), "%s",clave); }

    int32_t res = llama_model_meta_val_str(modelo, metaDato, buf, sizeof(buf)); 
    if (res <= 0) buf=0;
}

