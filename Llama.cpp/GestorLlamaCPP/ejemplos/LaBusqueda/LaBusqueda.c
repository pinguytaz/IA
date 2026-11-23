/***********************************************************************************
 * LaBusqueda Agente de busqueda, en el que le damos unos datos y realiza busqueda
 *            de información por internet y realiza al final un informe.
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 *	Modelos a utilizar:
 *
 * librerias:
 *	llama.cpp	Libreria que nos permite ganerar inferencia LLM en C
 ***********************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <curl/curl.h>
#include "llama.h"

#include "GestorLlamaCPP.h"
#include "funciones.h"
#include "funcIA.h"
#include "funcAgentes.h"
#include "funcLaBusqueda.h"

int main(int argc, char **argv) 
{
    if(argc != 2) { printf("Uso: %s \"<consulta>\"\n",argv[0]); return -1; }

    //***** Inicializaciones
    curl_global_init(CURL_GLOBAL_DEFAULT);   /* Una sola vez al arrancar */
    iniciaGestor(false);

    // Cargamos los roles
    arrayCadenas_t rolesSystem; 
    iniciaArrayCadenas(&rolesSystem, 4);  // Contendra el contenido de los roles
    char *fichRoles[] = { "../roles/RolConsulta.txt", 
                         "../roles/RolSintesis.txt", 
                         "../roles/RolInforme.txt", 
                         "../roles/RolCompara.txt",
                         NULL };
    cargaRoles(fichRoles, &rolesSystem);

    // Cargamos los modelos a usar
    bool error = false;
    //  Obtener consultas a la pregunta
    int modeloCon = cargaModelo("Qwen7B-instruct", "../modelos/qwen2.5-7b-instruct-q4_k_m.gguf");  
    if (modeloCon < 0) { error=true; } 

    // Realizar resumenes se usa también qwen 7B
    int modeloRes = modeloCon;
   
    // Generación de informes, dos modelos distintos
    int modeloIfA = cargaModelo("Mistral3 12B", "../modelos/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"); 
    if (modeloIfA < 0) { error=true; } 
    int modeloIfB = cargaModelo("Gemma2 9B", "../modelos/gemma-2-9b-it-Q4_K_M.gguf");
    if (modeloIfB < 0) { error=true; } 

    // Generación de analisis
    int modeloAnl = cargaModelo("Qwen 14B-instruct", "../modelos/qwen2.5-14b-instruct-q4_k_m.gguf");
    if (modeloAnl < 0) { error=true; } 

    // Si hay error en la carga se liberan recursos y salimos
    if (error)
    {
       fprintf(stderr, "Error: No se pudieron cargar todos los modelos.\n");
       // Liberación de recursos solicitados
       curl_global_cleanup(); 
       destructorGestor();
       return -1;
    }

    // **********  Definimos Samplers, para que puedan ser utilizados.
    paramSampler_t sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;   // En principio no medimos rendimiento
    // uint32_t semilla = (uint32_t)time(NULL);  // Aleatoria
    uint32_t semilla = 42; // La ponemos fija aseguramos uniformidad en desarrollo.
 
    // **** SAMPLER totalmente preciso "Greedy"
    sampler_t *samplerPreciso = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(samplerPreciso, llama_sampler_init_greedy()); // Mas estricta con temperatura 0
 
    // **** SAMPLER para Resumen 
    sampler_t *samplerResumen = llama_sampler_chain_init(sparams);
    //  CASTIGO(penalizaciones) --> PODA(K,P) --> Modelado(Temperatura --> Seleccion (distribucion con semilla)
    llama_sampler_chain_add(samplerResumen, llama_sampler_init_penalties(vocab_n_tokens(modeloRes), 128, 1.05f, 0.0f, 0.0f)); // VentanaTokens, repeticion exacta, frecuencia uso, presencia

    llama_sampler_chain_add(samplerResumen, llama_sampler_init_top_k(40));  // Tokens a considerar
    llama_sampler_chain_add(samplerResumen, llama_sampler_init_top_p(0.9f, 1)); // probabilidad y minimo a mantener 1.
    llama_sampler_chain_add(samplerResumen, llama_sampler_init_temp(0.3f));  // Temperatura controlada
    llama_sampler_chain_add(samplerResumen, llama_sampler_init_dist(semilla)); // Semilla fija

    // ***** SAMPLER para Informes, Como son modelos distintos y tiene penalización que depende del vocabulario del modelo se genera InfA e InfB
    sampler_t *samplerIfA = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(samplerIfA, llama_sampler_init_penalties(vocab_n_tokens(modeloIfA), 128, 1.1, 0.0f, 0.0f)); 
    llama_sampler_chain_add(samplerIfA, llama_sampler_init_top_k(40)); 
    llama_sampler_chain_add(samplerIfA, llama_sampler_init_top_p(0.95f, 1)); 
    llama_sampler_chain_add(samplerIfA, llama_sampler_init_temp(0.7f));     // Creativo 0.7 veremos si no se pasa.
    llama_sampler_chain_add(samplerIfA, llama_sampler_init_dist(semilla)); // Semilla fija

    sampler_t *samplerIfB = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(samplerIfB, llama_sampler_init_penalties(vocab_n_tokens(modeloIfB), 128, 1.1, 0.0f, 0.0f)); 
    llama_sampler_chain_add(samplerIfB, llama_sampler_init_top_k(40)); 
    llama_sampler_chain_add(samplerIfB, llama_sampler_init_top_p(0.95f, 1)); 
    llama_sampler_chain_add(samplerIfB, llama_sampler_init_temp(0.7f));     // Creativo 0.7 veremos si no se pasa.
    llama_sampler_chain_add(samplerIfB, llama_sampler_init_dist(semilla)); // Semilla fija

    // *****   SAMPLER para el analisis de comparación
    sampler_t *samplerAnalisis = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(samplerAnalisis, llama_sampler_init_penalties(vocab_n_tokens(modeloAnl), 128, 1.05, 0.0f, 0.0f)); 
    llama_sampler_chain_add(samplerAnalisis, llama_sampler_init_top_k(40)); 
    llama_sampler_chain_add(samplerAnalisis, llama_sampler_init_top_p(0.95f, 1)); 
    llama_sampler_chain_add(samplerAnalisis, llama_sampler_init_temp(0.5f));  // Analitica  
    llama_sampler_chain_add(samplerAnalisis, llama_sampler_init_dist(semilla)); // Semilla fija

    // ***** Comenzamos con el proceso  *****
    int retornoInferencia;

    //**** 1.- LLM1  Lo primero es realizar una inferencia para obtener consultas FUNC(busca:..)
    // **** Cambiamos algunos parametros del contexto, y tokes a recibirpara hacerlo un poco más pequeño
    paramContext_t paramCtx = get_ParamContexto(modeloCon);
    paramCtx.n_ctx = 3072;
    paramCtx.n_batch = 3072;
    paramCtx.n_ubatch = 512;
    put_ParamContexto(modeloCon, paramCtx);
    int numTokens = 225;

    StringDinamico_t respuestasCon;
    iniciaString(&respuestasCon); 
    retornoInferencia = inferir(modeloCon, argv[1] , rolesSystem.elementos[0], samplerPreciso, &respuestasCon,true,numTokens,VENTANA_DESLIZANTE);
    printf("Retorno Inferencia 1(Consultas) %d\n", retornoInferencia);
    printf("\tResultado[Capa:%lu Long:%lu]:\n%s\n",respuestasCon.capacidad, respuestasCon.longitud,respuestasCon.dato);
    printf("-------------------------------------\n");

    // **** 2.-FUNC Ejecutamos las funciones indicadas para la localización de URLs FUNC(url:)
    StringDinamico_t funcURL;
    iniciaString(&funcURL); 
    arrayCadenas_t respuestasURL; 
    iniciaArrayCadenas(&respuestasURL,3);  // URls de respuesta.
    procesarRespuestas((const char *)respuestasCon.dato, &respuestasURL); 
    printf("----------------------\n");
    int nRespuestasURLs = concatenaRespuestas(respuestasURL, &funcURL);
    printf("Total de resultados: %d \n",nRespuestasURLs);
    if(nRespuestasURLs == 0)   // No se encontrarron URLs nos vamos
    {
        printf("No he sido capaz de localizar URLs\n");

        // Limpiamos las cadenas dinamicas y array de cadenas
        liberaArrayCadenas(&rolesSystem);
        liberaArrayCadenas(&respuestasURL);
        limpiaString(&respuestasCon);
        limpiaString(&funcURL);
     
        // Limpia de Samplers creados
        llama_sampler_free(samplerPreciso);
        llama_sampler_free(samplerResumen);
        llama_sampler_free(samplerIfA);
        llama_sampler_free(samplerIfB);
        llama_sampler_free(samplerAnalisis);
     
        // Gestor e inicializaciones de CURL
        destructorGestor();
        curl_global_cleanup(); 
        return -2;
    }
    printf("Respuesta URLs:\n%s\n",funcURL.dato);
    printf("----------------------\n");

    // **** 3.- FUNC Obtenemos información de las URL dejando texto crudo
    arrayCadenas_t respuestasContenido; 
    iniciaArrayCadenas(&respuestasContenido,3);  // Contenido de las URLs
    procesarRespuestas((const char *)funcURL.dato, &respuestasContenido); // Localiza Información

    // **** 4.- LLM2  Realizamos resumenes y sintexis de los textos pasados en respuestasContenido a resumenes
    // **** Cambiamos algunos parametros del contexto, y tokes a recibirpara hacerlo un poco más pequeño
    paramCtx = get_ParamContexto(modeloRes);
    paramCtx.n_ctx = 6144;
    paramCtx.n_batch = 6144;
    paramCtx.n_ubatch = 512;
    put_ParamContexto(modeloRes, paramCtx);
    numTokens = 650;

    arrayCadenas_t resumenes; 
    iniciaArrayCadenas(&resumenes,3);    // Resumenes 

    StringDinamico_t preguntaRes;
    iniciaString(&preguntaRes); 
    ampliaString(&preguntaRes,1024);

    StringDinamico_t respuestaRes;
    iniciaString(&respuestaRes); 
    ampliaString(&respuestaRes,1024);

    int nResumenes=0;
    for(int i=0; i<respuestasContenido.nElementos; i++)
    {
        if(strcmp(respuestasContenido.elementos[i],"") != 0)
        {
            nResumenes++;
            preguntaRes.dato[0] = '\0';
            preguntaRes.longitud = 0;
            respuestaRes.dato[0] = '\0';
            respuestaRes.longitud = 0;
            
            printf("************************* [%d] Realiza inferencia *******************************\n",i); 
            cadenaAString(&preguntaRes, "Texto a resumir:\n");
            cadenaAString(&preguntaRes, respuestasContenido.elementos[i]);
            retornoInferencia = inferir(modeloRes, preguntaRes.dato , rolesSystem.elementos[1], samplerResumen, &respuestaRes,true,numTokens,VENTANA_DESLIZANTE);
            printf("Retorno Inferencia 2[%d](Resumen) %d\n", i, retornoInferencia);
            printf("\tResultado[Capacidad:%lu Longitud:%lu]:\n%s\n",respuestaRes.capacidad, respuestaRes.longitud,respuestaRes.dato);
            printf("-------------------------------------\n");
            addCadena(&resumenes, respuestaRes.dato);
        }
    }
    printf("Resumenes realizados %d\n",nResumenes);

    limpiaString(&preguntaRes);
    limpiaString(&respuestaRes);
    if(nResumenes == 0)   // No se encontrarron URLs nos vamos
    {
        printf("No he sido capaz de generar resumens\n");

        // Limpiamos las cadenas dinamicas y array de cadenas
        liberaArrayCadenas(&rolesSystem);
        liberaArrayCadenas(&respuestasURL);
        liberaArrayCadenas(&respuestasContenido);
        liberaArrayCadenas(&resumenes);
        limpiaString(&respuestasCon);
        limpiaString(&funcURL);
     
        // Limpia de Samplers creados
        llama_sampler_free(samplerPreciso);
        llama_sampler_free(samplerResumen);
        llama_sampler_free(samplerIfA);
        llama_sampler_free(samplerIfB);
        llama_sampler_free(samplerAnalisis);
     
        // Gestor e inicializaciones de CURL
        destructorGestor();
        curl_global_cleanup(); 
        return -2;
    }
    printf("Total de Resumenes: %d\n",nResumenes);


    // **** 5.- LLM3-4  Realizamos informes con los resumenes pasados, pero con dos modelos diferentes
    StringDinamico_t preguntaInf;
    iniciaString(&preguntaInf); 
    ampliaString(&preguntaInf,1024);
    preguntaInf.dato[0] = '\0';
    preguntaInf.longitud = 0;

    StringDinamico_t respuestaIfA;
    iniciaString(&respuestaIfA); 
    ampliaString(&respuestaIfA,1024);
    StringDinamico_t respuestaIfB;
    iniciaString(&respuestaIfB); 
    ampliaString(&respuestaIfB,1024);

    // Encabezado   === RESUMEN X ==== contenido === FIN RESUMEN X ====
     cadenaAString(&preguntaInf, "A continuación se presentan los resúmenes obtenidos de búsquedas independientes sobre el tema '");
     cadenaAString(&preguntaInf, argv[1]);
     cadenaAString(&preguntaInf, "' Consolida toda la información en un Informe Final.\n\n");
     for (int i=0; i< resumenes.nElementos; i++)
     {
         char tempo[12];
         snprintf(tempo,sizeof(tempo), "%d",i+1);
         cadenaAString(&preguntaInf, "==== RESUMEN ");
         cadenaAString(&preguntaInf, tempo);
         cadenaAString(&preguntaInf, " ==== \n ");

         cadenaAString(&preguntaInf, resumenes.elementos[i]);

         cadenaAString(&preguntaInf, "\n==== FIN RESUMEN ");
         cadenaAString(&preguntaInf, tempo);
         cadenaAString(&preguntaInf, " ==== \n ");
     }

     printf("Realizamos la pregunta %s\n a dos modelos diferentes", preguntaInf.dato);
     

     // **** Cambiamos algunos parametros del contexto, y tokes a recibirpara hacerlo un poco más pequeño
     numTokens = 2250;
     paramCtx = get_ParamContexto(modeloIfA);
     paramCtx.n_ctx = 8192;
     paramCtx.n_batch = 8192;
     paramCtx.n_ubatch = 512;
     put_ParamContexto(modeloIfA, paramCtx);
     retornoInferencia = inferir(modeloIfA, preguntaInf.dato , rolesSystem.elementos[2], samplerIfA, &respuestaIfA,true,numTokens,VENTANA_DESLIZANTE);
     printf("Retorno Inferencia 3(Informe A) %d\n", retornoInferencia);
     printf("\tResultado[Capacidad:%lu Longitud:%lu]:\n%s\n",respuestaIfA.capacidad, respuestaIfA.longitud,respuestaIfA.dato);
     printf("-------------------------------------\n");

     paramCtx = get_ParamContexto(modeloIfB);
     paramCtx.n_ctx = 8192;
     paramCtx.n_batch = 8192;
     paramCtx.n_ubatch = 512;
     put_ParamContexto(modeloIfB, paramCtx);
     retornoInferencia = inferir(modeloIfB, preguntaInf.dato , rolesSystem.elementos[2], samplerIfB, &respuestaIfB,true,numTokens,VENTANA_DESLIZANTE);
     printf("Retorno Inferencia 4(Informe B) %d\n", retornoInferencia);
     printf("\tResultado[Capacidad:%lu Longitud:%lu]:\n%s\n",respuestaIfB.capacidad, respuestaIfB.longitud,respuestaIfB.dato);
     printf("-------------------------------------\n");
 
     // **** 6.- LLM5  Realizamos el analisis de los dos informes anteriores
     StringDinamico_t preguntaAnl;
     iniciaString(&preguntaAnl); 
     ampliaString(&preguntaAnl,1024);
     preguntaAnl.dato[0] = '\0';
     preguntaAnl.longitud = 0;
 
     StringDinamico_t respuestaAnl;
     iniciaString(&respuestaAnl); 
     ampliaString(&respuestaAnl,1024);

     // Encabezado   === RESUMEN X ==== contenido === FIN RESUMEN X ====
     cadenaAString(&preguntaAnl, "A continuación se presentan dos informes generados sobre datos originales, genera la comparativa tecnica de ambos informes.\n");
     cadenaAString(&preguntaAnl, "==== INFORME DEL MODELO A (");
     cadenaAString(&preguntaAnl, get_NombreModelo(modeloIfA));
     cadenaAString(&preguntaAnl, ") ====\n");
     cadenaAString(&preguntaAnl, respuestaIfA.dato);
     cadenaAString(&preguntaAnl, "\n==== FIN INFORME DEL MODELO A ====\n");
     cadenaAString(&preguntaAnl, "==== INFORME DEL MODELO B (");
     cadenaAString(&preguntaAnl, get_NombreModelo(modeloIfB));
     cadenaAString(&preguntaAnl, ") ====\n");
     cadenaAString(&preguntaAnl, respuestaIfB.dato);
     cadenaAString(&preguntaAnl, "\n==== FIN INFORME DEL MODELO B ====\n");
     cadenaAString(&preguntaAnl, "Revisa bien los dos informes por separado antes de dar la valoración y con cual te quedarias\n");

    // **** Cambiamos algunos parametros del contexto, y tokes a recibirpara hacerlo un poco más pequeño
    numTokens = 2000;
    paramCtx = get_ParamContexto(modeloAnl);
    paramCtx.n_ctx = 8192;
    paramCtx.n_batch = 8192;
    paramCtx.n_ubatch = 512;
    put_ParamContexto(modeloAnl, paramCtx);
    retornoInferencia = inferir(modeloAnl, preguntaAnl.dato , rolesSystem.elementos[3], samplerAnalisis, &respuestaAnl,true,numTokens,VENTANA_DESLIZANTE);
    printf("Retorno Inferencia 5(Analisis informes) %d\n", retornoInferencia);
    printf("\tResultado[Capacidad:%lu Longitud:%lu]:\n%s\n",respuestaAnl.capacidad, respuestaAnl.longitud, respuestaAnl.dato);

    //***** Finalización y liberacion de recursos
    // Limpiamos las cadenas dinamicas y array de cadenas
    liberaArrayCadenas(&rolesSystem);
    liberaArrayCadenas(&respuestasURL);
    liberaArrayCadenas(&respuestasContenido);
    liberaArrayCadenas(&resumenes);
 
    limpiaString(&respuestasCon);
    limpiaString(&funcURL);
    limpiaString(&preguntaInf);
    limpiaString(&respuestaIfA);
    limpiaString(&respuestaIfB);
    limpiaString(&preguntaAnl);
    limpiaString(&respuestaAnl);
 
    // Limpia de Samplers creados
    llama_sampler_free(samplerPreciso);
    llama_sampler_free(samplerResumen);
    llama_sampler_free(samplerIfA);
    llama_sampler_free(samplerIfB);
    llama_sampler_free(samplerAnalisis);

    // Gestor e inicializaciones de CURL
    destructorGestor();
    curl_global_cleanup(); 
    printf("************ Esto es todo amigos ******************\n\n");

}
