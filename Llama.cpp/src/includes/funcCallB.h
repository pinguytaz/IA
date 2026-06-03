/***********************************************************************************
 * Libreria que contendra las funciones CallBack a utilizar
 *
 * Autor: Fco. Javier Rodriguez Navarro
 * Web:   https:/www.pinguytaz.net
 * gihub: https://github.com/pinguytaz
 *
 ***********************************************************************************/
#ifndef FUNCCALLB_H
#include "llama.h"
#define FUNCCALLB_H

/**
 * @details Esta función, CallBack recoge info para el log pero no lo imprime.
 * 
 * @param nivel del log
 * @param texto del log
 * @param datos usuario
 * @return nada
 */
void mi_funcion_log_silenciosa(enum ggml_log_level level, const char * text, 
		               void * user_data) 
{
    // No hacemos nada aquí. Al dejarla vacía, bloqueamos el texto.
}


/**
 * @details Esta función, es CallBack imprime el proceso de carga del modelo
 * 
 * @param porcentaje del proceso
 * @param datos usuario
 * @return true para continuar cargando o false se corta
 */
bool mi_progreso(float progreso, void *datosUsuario) 
{
    printf("\rCargando modelo: %.1f%%", progreso * 100);
    fflush(stdout); // Para que se vea en tiempo real
    return true; // Continuar cargando
}

#endif
