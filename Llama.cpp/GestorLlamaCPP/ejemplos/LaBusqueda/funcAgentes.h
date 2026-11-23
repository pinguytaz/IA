
/*********************************************************************************
 * Analizador y funciones caja Herramientaszando funciones
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 ********************************************************************************/
#ifndef FUNCAGENTES
#include "funciones.h"
#define FUNCAGENTES

/*********************************************************************************/
// Toda funcion de accion de agente tendra esta forma
typedef void (*AgenteAccion)(const char *params, char *resultado, int lenResultado);

//  Estructura de la herramienta: nombre funcion y argumentos
typedef struct
{
    const char *nombre;
    AgenteAccion ejecutar;
} Herramienta_t;

int procesarRespuestas(const char *texto_ia, arrayCadenas_t *resultado);

// Definicion las herramientas de la caja
void cmdBusca(const char *p, char *resultado, int maxResultado) ;
void cmdURL(const char *p, char *resultado, int maxResultado) ;
//void cmdReduce(const char *p, char *resultado, int maxResultado) ;

#endif
