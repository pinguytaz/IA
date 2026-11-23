/************************************************************************
 * Funciones de proposito generar, auxiliares para desarrollos
 * y para el Gestor pero que se puede usar perfectamente en otros proyectos
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 * Tipos
 *	array_Cadenas_t    Nos permite contener un array de strings
 *      
 * cargaFichero      Carga un fichero de texto que lo salva en una variable
 *
 ***************************************************************************/

#ifndef FUNCIONES_H
#include <stddef.h>

#define FUNCIONES_H

////////////////////////// Estructuras y tipos  //////////////////////////////////////////////
// **** Estructura gestion Array dinamico de cadenas
typedef struct 
{
    char **elementos;     // El array de punteros a cadenas (char**)
    size_t nElementos;    // Número actual de elementos guardados
    size_t capacidad;     // Capacidad máxima actual antes de necesitar realloc
} arrayCadenas_t;

////////  Funciones array de cadenas   /////////
int iniciaArrayCadenas(arrayCadenas_t *arr, size_t capacidadInicial);
int addCadena(arrayCadenas_t *arr, const char *cadena);
void liberaArrayCadenas(arrayCadenas_t *arr);
size_t numeroElementos(const arrayCadenas_t *arr);


// **** Estructura para mantener un String dinamico
typedef struct 
{
    char *dato;
    size_t capacidad;   // Tiene la capacidad maxima, para saber si debe aumentarla
    size_t longitud;    // Longitud del array sin el '\0' asi saber donde se van añadiendo datos
} StringDinamico_t;

void iniciaString(StringDinamico_t *str);
void limpiaString(StringDinamico_t *str) ;
void ampliaString(StringDinamico_t *str, int incremento) ;
void cadenaAString(StringDinamico_t *str, char *cadena) ;


/////////////////////////////////////////////////////////////////////////////////////////////


/////////////// Funciones
int cargaFichero(char *pathFichero, char *respuesta, size_t maxRespuesta);



#endif
