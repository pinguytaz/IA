/*********************************************************************************
 * Funciones para analisis respuesta localizando funciones
 * Y las funciones agenticas
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 ********************************************************************************/
#ifndef FUNCAGENTES
#include <regex.h>
#include <stdio.h>
#include <string.h>
#include "CajaHerramientas.h"
#define FUNCAGENTES

// Toda funcion de accion de agente tendra esta forma
typedef void (*AgenteAccion)(const char *params);


//  Estructura de la herramienta: nombre funcion y argumentos
typedef struct 
{
    const char *nombre;
    AgenteAccion ejecutar;
} Herramienta_t;


// ARRAY DE HERRAMIENTAS (Tu "caja de herramientas")
Herramienta_t catalogo[] = 
{
   {"elTiempo",   cmd_elTiempo},
   {"laFecha",  cmd_laFecha}
};
#define NUM_HERRAMIENTAS (sizeof(catalogo) / sizeof(Herramienta_t))


/**
 * @details Realiza el procesado del texto localizando FUNC e invocando
 * 
 * @param texto que nos ha dado el modelo LLM que tendra FUNC(---)
 * @return nada
 */
void procesarRespuestas(const char *texto_ia) 
{
   regex_t regex;
   regmatch_t matches[3];

   /* Patron 
	FUNC\\( ([^:)]+) :? ([^)]*) \\) 
	^   	^  	 ^  ^       ^  
	|   	|  	 |  |       |
	|   	|  	 |  |       L----Cierre ")"
	|       |        |  L------------Grupo captura parametros opcional fin ")"
	|   	|  	 L---------------":" opcional
	|	L-------------------------Grupo captura función hasta ":"
	L---------------------------------Apertura "FUNC("
   */
   const char *patron = "FUNC\\(([^:)]+):?([^)]*)\\)";

   // Pasamos el patron con el motor REG_EXTENDED (Extended Regular Expressions)
   if (regcomp(&regex, patron, REG_EXTENDED) != 0) return;

   // Tenemos funciones encontradas
   const char *cursor = texto_ia;
   while (regexec(&regex, cursor, 3, matches, 0) == 0) 
   {
      char func_nombre[64] = {0};
      char params[256] = {0};

      // rm_so (Start Offset): El índice del carácter donde empieza la coincidencia.
      // rm_eo (End Offset): El índice del carácter donde termina la coincidencia 
      // patron completo encontrado
     /*
      int inicio = matches[0].rm_so;
      int fin    = matches[0].rm_eo;
      int longitud = fin - inicio;
      char nombre[64];
      strncpy(nombre, cursor + inicio, longitud);
      nombre[longitud] = '\0'; // ¡Importante!
      printf("Encontramos-->%s\n",nombre);
     */ 
      // Extraer Nombre
      int len_f = matches[1].rm_eo - matches[1].rm_so;
      strncpy(func_nombre, cursor + matches[1].rm_so, len_f);

      // Extraer Parámetros (si los hay)
      int len_p = matches[2].rm_eo - matches[2].rm_so;
      if (len_p > 0)  // Si tiene paramentros
      {
         strncpy(params, cursor + matches[2].rm_so, len_p);
      }

      // Ejecutamos
      //printf("\tfuncion->%s y Parametros: %s\n",func_nombre, params);
      bool encontrada = false;
      for (int i = 0; i < NUM_HERRAMIENTAS ; i++)
      {
         if (strcmp(func_nombre, catalogo[i].nombre) == 0) 
	 {
	    encontrada = true;
            catalogo[i].ejecutar(params);
            break;
        }
      }
      if(!encontrada) printf("No implementada %s parametros %s\n",func_nombre,params);

      cursor += matches[0].rm_eo; // Avanzamos a la siguiente encontrada
    }
    regfree(&regex);  // Libera estructura
}

#endif
