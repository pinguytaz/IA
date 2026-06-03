/***********************************************************************************
 * funciones utilizada para pruebas que nos permite visualizar datos
 * de los modelos, etc.
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 ***********************************************************************************/
#ifndef FUNCPRUEBAS_H
#include "llama.h"
#include "../includes/tiposLlamacpp.h"
#define FUNCPRUEBAS_H


/**
 * @details Realiza un volcado de las etiquetas del modelo, para verlas
 * 
 * @param modelo
 * @return nada
 */
void volcado_MetaModelo(modelo_t *modelo)
{
    char clave[256]; 
    char buffer_meta[512]; 
    int32_t res;
    int32_t paregas=llama_model_meta_count(modelo);
    printf("numero de claves %d\n", paregas);
       
    for (int32_t i=0; i<paregas; i++)
    {
        llama_model_meta_key_by_index(modelo, i, clave, sizeof(clave));
        res = llama_model_meta_val_str(modelo, clave, buffer_meta, sizeof(buffer_meta));
        if (res > 0) printf("%d %s: %s \n", i, clave, buffer_meta);
    }
}


/**
 * @details Da la etiqueta solicitada del modelo
 * 
 * @param modelo
 * @param clave a buscar
 * @param donde se guardara la información obtenida.
 * @return nada
 */
void daEtiquetaMetaModelo(modelo_t *modelo, const char *clave, char *buf)
{
    int32_t res = llama_model_meta_val_str(modelo, clave, buf, sizeof(buf)); 
    if (res <= 0) buf=0; 
}

/**
 * @details Imprime la información del modelo pasado
 * 
 * @param modelo
 * @return nada
 */
void infoModelo(modelo_t *modelo)
{
    // Obtener el nombre del modelo o descripción que le puso el autor
    char buffer_meta[512]; 
    // Metadatos del modelo
    //volcado_MetaModelo(modelo);   // Visualiza etitqetas Meta
    daEtiquetaMetaModelo(modelo, "general.name", buffer_meta); 
    printf("\n\tNombre: %s", buffer_meta);
    daEtiquetaMetaModelo(modelo, "general.basename", buffer_meta); 
    printf(" Familia: %s", buffer_meta);
    daEtiquetaMetaModelo(modelo, "general.architecture", buffer_meta); 
    printf(" Arquitectura: %s\n", buffer_meta);
    daEtiquetaMetaModelo(modelo, "llama.context_length", buffer_meta); 
    printf("\tTamaño del contexto: %s\n", buffer_meta);
    daEtiquetaMetaModelo(modelo, "llama.block_count", buffer_meta); 
    printf("\tCapas RedNeuronal: %s\n", buffer_meta);
    daEtiquetaMetaModelo(modelo, "tokenizer.ggml.bos_token_id", buffer_meta); 
    printf("\tTOKE BOS: %s", buffer_meta);
    daEtiquetaMetaModelo(modelo, "tokenizer.ggml.eos_token_id", buffer_meta); 
    printf("\tTOKEN EOS: %s\n", buffer_meta);

    //  Tamaño del modelo 
    uint64_t bytes_modelo = llama_model_size(modelo);
    double gb_modelo = ((double)bytes_modelo / (1024.0 * 1024.0)) / 1024.0;
    printf("\tTamaño del modelo %lu bytes / %.0f Gigas\n", bytes_modelo,gb_modelo);
    // 2. Obtener número de parámetros
    uint64_t parametros = llama_model_n_params(modelo);
    printf("\tNumero de parametros: %lu \n", parametros);
}


/**
 * @details Imprimimos los TOKENs del prompt pasado.
 * 
 * @param vocabulario del modelo
 * @param Numero de tokens
 * @param Array de Tokens con el prompt
 * @return nada
 */
void desgloseTOKENS(const vocab_t *vocab, int32_t n_tokens, llama_token *prompt_tokens)
{
    printf("\n--- Desglose del Vocabulario para el Prompt ---\n");
    char token_piece[64];
    for (int i = 0; i < n_tokens; i++) 
    {
        int32_t bytes_escritos = llama_token_to_piece(vocab,
        prompt_tokens[i],
        token_piece,
        sizeof(token_piece),
        0,     // lstrip: 0 para mantener espacios iniciales si el modelo los tiene
        true   // special: true para que pinte textos como <|begin_of_text|> si es un token de control
        );

        if (bytes_escritos > 0) 
	{
            // Aseguramos que la cadena termine en el carácter nulo de C por seguridad
            token_piece[bytes_escritos] = '\0';

            // Imprimimos el ID y el trozo de texto entre corchetes para ver los espacios ocultos
            printf("Token %d -> ID: %6d | Fragmento: [%s]\n", i, prompt_tokens[i], token_piece);
        }
    }
}


/**
 * @details Información de un lote de prompts
 * 
 * @param Lote
 * @return nada
 */
void infoBatch(llama_batch batch)
{
    printf("\tNumero de tokens: %d",batch.n_tokens);
    printf("\tTokens: [ ");
    for (int i=0; i< batch.n_tokens; i++)
    {
        printf("%d",batch.token[i]);
	if(i != batch.n_tokens-1) printf(", ");
    }
    printf("]\n");

    if (batch.logits == NULL) printf("\tLogits es nulo\n");

}

#endif
