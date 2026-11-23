/* Sencillo ejemplo de uso */

#include <time.h>
#include "GestorLlamaCPP.h"
#include "funciones.h"

int main(int argc, char **argv) 
{
    if(argc != 4) { printf("Uso: %s <modelo> <ROL-System> <consulta>\n",argv[0]); return -1; }

    // modelo argv[1], fichero rol argv[2] y pregunta argv[3]

   char rolSystem[1024];
   cargaFichero(argv[2], rolSystem, sizeof(rolSystem));

  iniciaGestor(false);
  // Cargamos el modelo pasado por parametro
  int modelo = cargaModelo("Nombre del modelo", argv[1]);
  // **** Cambiamos algunos parametros del contexto, para hacerlo un poco más pequeño
  paramContext_t paramCtx = get_ParamContexto(modelo);
  paramCtx.n_ctx = 2048;
  paramCtx.n_batch = 2048;
  paramCtx.n_ubatch = 512;
  put_ParamContexto(modelo, paramCtx);

  listaModelos();

  // ********************* SAMPLER ********************************
  // Creamos un sampler preciso temperatura 0. y uno muy creativo de mas temperatura
  paramSampler_t sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;   // En principio no medimos rendimiento
  sampler_t *samplerPreciso = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(samplerPreciso, llama_sampler_init_greedy()); // Mas estricta con temperatura 0

  // Creativo
  sampler_t *samplerCreativo = llama_sampler_chain_init(sparams);
  // MUY recomendado para Qwen, Limitar el espacio de muestreo
  // Evitamos que el modelo asigne probabilidad a tokens basura que pueden causar NaN
  llama_sampler_chain_add(samplerCreativo, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(samplerCreativo, llama_sampler_init_top_p(0.9f, 1));

  llama_sampler_chain_add(samplerCreativo, llama_sampler_init_temp(1.5f)); // Temperatura

  // ¡CRÍTICO! Añadir el muestreador de distribución que SÍ selecciona el token.
  uint32_t seed = (uint32_t)time(NULL); // Semilla aleatoria
  llama_sampler_chain_add(samplerCreativo, llama_sampler_init_dist(seed));


  // Lanzamos la inferencia, mismo modelo, misma pregunta pero con dos sampler distintos.
  int maxTokens = 500;

  // Creamos un String Dinamico de respuesta 
  StringDinamico_t respuesta;
  iniciaString(&respuesta); 

  inferir(modelo, argv[3], rolSystem, samplerPreciso, &respuesta,true,maxTokens,VENTANA_DESLIZANTE);
  printf("Resultado con preciso :\n%s\n",respuesta.dato);
  printf("\n\nLongitud de la respuesta precisa:\n\t%zu\n\tCapacidad: %zu\n",respuesta.longitud,respuesta.capacidad);
  printf("-------------------------------------\n");

  inferir(modelo, argv[3], rolSystem, samplerCreativo, &respuesta,false,maxTokens,VENTANA_DESLIZANTE); 
  printf("Resultado con Creativo :\n%s\n",respuesta.dato);
  printf("\n\nLongitud de la respuesta Creativa:\n\t%zu\n\tCapacidad: %zu\n",respuesta.longitud,respuesta.capacidad);
  printf("-------------------------------------\n");


  llama_sampler_free(samplerCreativo);
  llama_sampler_free(samplerPreciso);

  limpiaString(&respuesta);
  destructorGestor();
  return 0;
}
