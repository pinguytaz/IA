#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "funciones.h"

////////////   FUNCIONES DE STRINGs DINAMICOS ////////////////////////////////////////////
//  Inicializa string dinamico sin datos ni capacidad
void iniciaString(StringDinamico_t *str)
{
    str->dato = NULL; 
    str->capacidad = 0;
    str->longitud = 0;
}    
// libera el string dinamico quedando como nuevo para liberar memoria.i hacerlo al final de programa
void limpiaString(StringDinamico_t *str) 
{
    if (str->dato) 
    {
        free(str->dato);
        str->dato = NULL;
    }
    str->capacidad = 0;
    str->longitud = 0;
}
// amplia la capacidad del String
void ampliaString(StringDinamico_t *str, int incremento) 
{
    int nuevaCapacidad = str->capacidad + incremento;
    char *temp = (char *)realloc(str->dato, nuevaCapacidad);
    str->dato = temp;
    str->capacidad = nuevaCapacidad;
}
// Añade cadena
void cadenaAString(StringDinamico_t *str, char *cadena) 
{
    int longitudCadena = strlen(cadena);
    if(str->capacidad < longitudCadena + str->longitud){ ampliaString(str,longitudCadena + str->longitud + 10);}
    memcpy(str->dato + str->longitud, cadena, longitudCadena);
    str->longitud += longitudCadena;
    str->dato[str->longitud] = '\0';
}

/////////////////////////////////////////////////////////////////////7////////////////////

////////////   FUNCIONES DE ARRAY de CADENAS ////////////////////////////////////////////
/** duplicador de cadena para que no se pierdan ********************** */
static char* duplicaCadena(const char *cadena) 
{
    if (cadena == NULL) return NULL;
    size_t len = strlen(cadena) + 1;
    char *dup = malloc(len);
    if (dup != NULL) { memcpy(dup, cadena, len); }
    return dup;
}

/********************* Inicializa el array de cadenas****************************/
int iniciaArrayCadenas(arrayCadenas_t *arr, size_t capacidadInicial) 
{
    arr->nElementos = 0;    // 0 elementos inicialmente
    if (capacidadInicial <= 0) capacidadInicial = 1; //Evitamos crearlo a 0 de capacidad.
    arr->capacidad = capacidadInicial;
    arr->elementos = malloc(arr->capacidad * sizeof(char *));
    if (arr->elementos == NULL) {fprintf(stderr, "Error al asignar memoria en array de Cadenas\n"); return -1; }
    return 0;
}

/***********  añade una cadena ****************************/
int addCadena(arrayCadenas_t *arr, const char *cadena) 
{
   // Si el array se ha llenado, duplicamos su capacidad
   if (arr->nElementos >= arr->capacidad) 
   {
      arr->capacidad *= 2;
      char **temp = realloc(arr->elementos, arr->capacidad * sizeof(char *));
      if (temp == NULL) {fprintf(stderr, "Error al redimensionar memoria en array de Cadenas\n"); return -1; }
      arr->elementos = temp;
   }

   // Duplicamos la cadena de entrada para asegurar que tenga su propia memoria reservada
   arr->elementos[arr->nElementos] = duplicaCadena(cadena);
   if (arr->elementos[arr->nElementos] == NULL) {fprintf(stderr, "Error al duplicar memoria en array de Cadenas\n"); return -1; }

   arr->nElementos++;
   return 0;
}

/******** liberamos memoria del array de cadenas, no olvidar al finalizar para evitar fugas "memory leaks" ***/
void liberaArrayCadenas(arrayCadenas_t *arr) 
{
   // Primero liberamos las cadenas individualmente.
   for (size_t i = 0; i < arr->nElementos; i++) { free(arr->elementos[i]); }
   free(arr->elementos);        // Liberamos el array de punteros
   arr->nElementos = 0;
   arr->capacidad = 0;
}

/********** Nos da el numero de elementos guardados *************/
size_t numeroElementos(const arrayCadenas_t *arr) { return arr->nElementos;  }

//////////////////////////// Fin funciones tratamiento array cadenas ////////////////////////////////////



/************************************************************************
 * Carga de fichero de tecto y se pasa a variable
 *
 * Parametros: 
 *    fichero nombre del fichero
 *    contenido contenido del fichero 
 *    maxContenido Maximo tamaño a pasar.
 *
 * Retorno negativo, error, positivo el tamaño descargado.
 ***************************************************************************/
int cargaFichero(char *pathFichero, char *respuesta, size_t maxRespuesta)
{
   respuesta[0] =0;
   FILE *f = fopen(pathFichero, "rb");
   if (!f) { fprintf(stderr, "Error: cargando Fichero %s \n", pathFichero); return -1; }
   fseek(f, 0, SEEK_END);  // Al final para ver tamaño
   long tam = ftell(f);  // Tamaño del fichero

   rewind(f);  // Volvemos al inicio
   if(tam > maxRespuesta - 1) { fprintf(stderr, "Error: tamaño del Fichero %s muy grande\n", pathFichero); return -2; }
   
   size_t leidos = fread(respuesta, 1, (size_t)tam, f);
   respuesta[leidos] = '\0';
   fclose(f);
   return tam;
}

