#include <stddef.h>
#include <stdio.h>
#include "funciones.h"

/************************************************************************
 * Cargamos roles, que se usaran durante el programa 
 *
 * Parametros: 
 *    ficheros  array de ficheros de roles
 *    datos     array de cadenas con los datos capturadoscontenido del fichero 
 *
 * Retorno negativo, error, positivo el numero de roles descargados
 ***************************************************************************/
int cargaRoles(char **pathFicheros, arrayCadenas_t *resultado)
{

   int contador = 0;
   while (pathFicheros[contador] != NULL) 
   {
      char contenido[2048];
      cargaFichero(pathFicheros[contador],contenido,sizeof(contenido)); 
      addCadena(resultado, contenido);
      contador++;
    }
   return 0;
}
