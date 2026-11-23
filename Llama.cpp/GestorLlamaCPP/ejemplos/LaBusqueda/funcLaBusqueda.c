#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "funciones.h"
#include "funcLaBusqueda.h"


/***  
   Recoge respuestas de un arry de cadenas y las va concatenando todas en 
   un unico resultado de StringDinamico
   Si no tenemos ninguna respuesta retorna 0 (no hay datos), sino el numero 
   de respuestas con datos.
******/
int concatenaRespuestas(arrayCadenas_t respuestas, StringDinamico_t *resultado)
{
    int conDatos=0;
    //if (resultado->dato == NULL) // Creamos con capacidad inicial de AMPLIACIONES.
    //{
         //char *temp = (char *)realloc(resultado->dato, AMPLIACIONES+1);
         //resultado->dato = temp;
         //resultado->capacidad = AMPLIACIONES;
    //} 
    //resultado->dato[0] = '\0';
    //resultado->longitud=0;
    
     //printf("INICIAL capacidad: %lu, Longitud: %lu\n",resultado->capacidad, resultado->longitud);
    // Iniciamos resultado a vacio
    for(int i=0; i<respuestas.nElementos; i++)
    {
        if(strcmp(respuestas.elementos[i],"") != 0) // Tenemos respuestas
        {
            conDatos++;
            cadenaAString(resultado, respuestas.elementos[i]);

            //size_t longDato = strlen(respuestas.elementos[i]) ;
            //size_t espacio_necesario = longDato + 1; // +1 para el '\0'
            //size_t espacio_disponible = resultado->capacidad - resultado->longitud;

            //printf("[%d] Tenemos %d datos LONG %ld ----> %s\n",i,conDatos,longDato, respuestas.elementos[i]);
            //printf("Necesita %ld Disponible %ld \n",espacio_necesario,espacio_disponible);
            //if (espacio_necesario > espacio_disponible)
            //{
                // Calcular cuántos chunks necesitamos, de 10 caracters
                //size_t deficit = espacio_necesario - espacio_disponible;
                //size_t chunks_adicionales = (deficit + AMPLIACIONES - 1) / AMPLIACIONES;
                //size_t expansion = chunks_adicionales * AMPLIACIONES; 
                //printf("EXPANDE: %lu\n",expansion);
                //char *temp = (char *)realloc(resultado->dato, expansion+resultado->capacidad);
                //if (!temp) {printf("Error asignando memoria");}
                //resultado->capacidad += expansion;
                //resultado->dato = temp;
            //}
            //printf("Actuala SCAD capacidad: %lu, Longitud: %lu\n",resultado->capacidad, resultado->longitud);

            // Copiamos fragmento
            //printf("Añadiendo[%ld] --->%s\n",longDato, respuestas.elementos[i]);

            //memcpy(resultado->dato + resultado->longitud, respuestas.elementos[i], longDato);
            //resultado->longitud += longDato;
            //resultado->dato[resultado->longitud] = '\0';
            //printf("FIN  capacidad: %lu, Longitud: %lu\n",resultado->capacidad, resultado->longitud);
        } 
     }
     return conDatos;
} 
