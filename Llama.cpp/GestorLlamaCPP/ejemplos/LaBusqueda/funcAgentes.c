#include <regex.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <curl/curl.h>
#include <cJSON.h>
#include <libxml/HTMLparser.h>
#include <libxml/tree.h>

#include "funcAgentes.h"

// ARRAY DE HERRAMIENTAS (Tu "caja de herramientas")
Herramienta_t catalogo[] =
{
   {"busca",   cmdBusca},
   {"url",   cmdURL}
};


#define NUM_HERRAMIENTAS (sizeof(catalogo) / sizeof(Herramienta_t))

/* ------------------------------------------------------------------
 * Estructura auxiliar para acumular la respuesta HTTP.
 * libcurl entrega los datos por fragmentos mediante un callback;
 * necesitamos un buffer que crezca dinámicamente.
 * ------------------------------------------------------------------ */
typedef struct 
{
    char  *data;   /* Puntero al buffer */
    size_t size;   /* Tamaño actual (sin contar el '\0') */
} ResponseBuffer;


/**
 * @details Procesa funciones FUNC de la cadena y retorna resultados en array
 * 
 * @param texto que nos ha dado el modelo LLM que tendra FUNC(---)
 * @return nada
 */
int procesarRespuestas(const char *texto_ia, arrayCadenas_t *resultado) 
{
   regex_t regex;
   regmatch_t matches[3];
   int ejecuciones = 0; // Inicialmente no hay ninguna ejecucion

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
   if (regcomp(&regex, patron, REG_EXTENDED) != 0) 
   {
      regfree(&regex);
      return ejecuciones ;
   }

   // Tenemos funciones encontradas
   const char *cursor = texto_ia;
   while (regexec(&regex, cursor, 3, matches, 0) == 0) 
   {
      char func_nombre[64] = {0};
      char params[256] = {0};

      // Extraer Nombre
      int len_f = matches[1].rm_eo - matches[1].rm_so;
      strncpy(func_nombre, cursor + matches[1].rm_so, len_f);

      // Extraer Parámetros (si los hay)
      int len_p = matches[2].rm_eo - matches[2].rm_so;
      if (len_p > 0) { strncpy(params, cursor + matches[2].rm_so, len_p); }

      // Ejecutamos
      //printf("\tfuncion->%s y Parametros: %s\n",func_nombre, params);
      bool encontrada = false;
      for (int i = 0; i < NUM_HERRAMIENTAS ; i++)
      {
         if (strcmp(func_nombre, catalogo[i].nombre) == 0)
         {
            encontrada = true;
            ejecuciones++;
            char res_func[50 * 1024];
            catalogo[i].ejecutar(params,res_func,sizeof(res_func));
            //printf("\trespuesta->%s\n",res_func);
            // añadimos resultado al Array
            addCadena(resultado, res_func);
            break ;
         } // If comprueba función con exito
      } // FOR
      if(!encontrada) printf("No implementada %s parametros %s\n",func_nombre,params);
      cursor += matches[0].rm_eo; // Avanzamos a la siguiente encontrada
   } // WHILE

   regfree(&regex);
   return ejecuciones;
}

/* ------------------------------------------------------------------
 * Si es una URL externa la convierte a FUNC(url:...) 
 * Evita duplicados y filtra enlaces internos de DuckDuckGo.
 *
 * Parametros:
 * 	url    URL completa (http:// o https://)
 * 	resultado
 * 	capacidad
 * 	contador
 * ------------------------------------------------------------------ */
static void agregarComandoURL(const char *url, char *resultado, int maxResultado, int *pos) 
{
    if (!url || strlen(url) == 0) return;
    
    /* Filtrar URLs de redirección o internas de DuckDuckGo */
    if (strstr(url, "duckduckgo.com") != NULL) return;

    /* Formatear exactamente tipo FUNC para poder las procesar */
    char linea[512];
    snprintf(linea, sizeof(linea), "FUNC(url:%s)\n", url);

    int espacio_restante = maxResultado - *pos; 
    if (espacio_restante > 1)
    {
       int escritos = snprintf(resultado + *pos, espacio_restante, "%s", linea);
       if (escritos > 0) { *pos += (escritos < espacio_restante) ? escritos : (espacio_restante - 1); }
    }
}



/* ------------------------------------------------------------------
 * Extrae URLs y las retorna como FUNC(url:..)
 *
 * Parametros:
 * 	json_str  Cadena de texto con JSON, Retorno de busqueda
 * 	resultado  Cadena de texto con las FUNC de cada url
 * 	lenResultado Maximo tamaño de la cadena resultado
 * ------------------------------------------------------------------ */
static void obtenerURLsJSON(const char *json_str, char *resultado, int lenResultado) 
{
    int pos = 0;
    resultado[0]='\0';
    if (!json_str || strlen(json_str) == 0) { return ; }

    cJSON *root = cJSON_Parse(json_str);
    if (!root) { return ; }

    /* 1. Buscar AbstractURL (fuente principal) */
    cJSON *abstract_url = cJSON_GetObjectItem(root, "AbstractURL");
    if (cJSON_IsString(abstract_url) && abstract_url->valuestring) { agregarComandoURL(abstract_url->valuestring, resultado, lenResultado, &pos); } 
    else if (cJSON_IsArray(abstract_url)) 
    {
        cJSON *item;
        cJSON_ArrayForEach(item, abstract_url) 
        {
            if (cJSON_IsString(item) && item->valuestring) 
            {
                agregarComandoURL(item->valuestring, resultado, lenResultado, &pos);
            }
        }
    }

    /* 2. Buscar FirstURL en Results (resultados directos) */
    cJSON *results = cJSON_GetObjectItem(root, "Results");
    if (cJSON_IsArray(results)) 
    {
        cJSON *item;
        cJSON_ArrayForEach(item, results) 
        {
            cJSON *first_url = cJSON_GetObjectItem(item, "FirstURL");
            if (cJSON_IsString(first_url) && first_url->valuestring) {
                agregarComandoURL(first_url->valuestring, resultado, lenResultado, &pos);
            }
        }
    }

    /* 3. Buscar FirstURL en RelatedTopics (temas relacionados) */
    cJSON *related = cJSON_GetObjectItem(root, "RelatedTopics");
    if (cJSON_IsArray(related)) 
    {
        cJSON *item;
        cJSON_ArrayForEach(item, related) 
        {
            /* Caso A: URL directa en el tema */
            cJSON *first_url = cJSON_GetObjectItem(item, "FirstURL");
            if (cJSON_IsString(first_url) && first_url->valuestring) 
            {
                agregarComandoURL(first_url->valuestring, resultado, lenResultado, &pos);
            }
            
            /* Caso B: Sub-temas anidados (Topics dentro de RelatedTopics) */
            cJSON *subtopics = cJSON_GetObjectItem(item, "Topics");
            if (cJSON_IsArray(subtopics)) 
            {
                cJSON *subitem;
                cJSON_ArrayForEach(subitem, subtopics) 
                {
                    cJSON *sub_first_url = cJSON_GetObjectItem(subitem, "FirstURL");
                    if (cJSON_IsString(sub_first_url) && sub_first_url->valuestring) 
                    {
                        agregarComandoURL(sub_first_url->valuestring, resultado, lenResultado, &pos);
                    }
                }
            }
        }
    }

    cJSON_Delete(root);

    return ;
}


/* ------------------------------------------------------------------
 * Funcion CallBack que se invoca cada vez que llega un fragmento
 *
 * Parametros:  contents Datos size  ram nmemb  userp 
 *
 * Retorno  0 error, sino el numero de bytes de respuesta.
 *          NULL en caso de error (se imprime diagnóstico por stderr).
 * ------------------------------------------------------------------ */
static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) 
{
    size_t total = size * nmemb;
    ResponseBuffer *buf = (ResponseBuffer *)userp;

    char *tmp = realloc(buf->data, buf->size + total + 1);
    if (!tmp) return 0;   // Fallo de memoria: señalamos error a curl 
    buf->data = tmp;

    memcpy(&(buf->data[buf->size]), contents, total);
    buf->size += total;
    buf->data[buf->size] = '\0';  // Mantenemos el buffer siempre nul-terminado 
    return total;
}


/* ------------------------------------------------------------------
 * Descarga el contenido HTML de una URL.
 *
 * Parametros:
 * 	url    URL completa (http:// o https://)
 * 	max_bytes     Tamaño máximo a descargar (0 = sin límite práctico)
 *
 * retorno  Puntero a string con el HTML. El llamador debe liberarlo con free().
 *          NULL en caso de error (se imprime diagnóstico por stderr).
 * ------------------------------------------------------------------ */
static char *peticionURL(const char *url, size_t max_bytes)
{
    //printf("*** Peticion URL: %s \n", url);

    if (!url || !*url) {
        fprintf(stderr, "peticionURL: URL vacía o NULL.\n");
        return NULL;
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "peticionURL: no se pudo inicializar libcurl.\n");
        return NULL;
    }

    /* Buffer de respuesta, inicializado vacío. */
    ResponseBuffer buf = { .data = calloc(1, 1), .size = 0 };
    if (!buf.data) {
        curl_easy_cleanup(curl);
        return NULL;
    }

    /* 1. Configuración básica */
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buf);

    /* User-Agent de navegador real */
    curl_easy_setopt(curl, CURLOPT_USERAGENT,
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36");

    /* 2. Configuración SSL ESTRICTA Y CORRECTA */
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_CAINFO, "/etc/ssl/certs/ca-certificates.crt");
    curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);

    /* 3. Evasión de Firewalls (WAF) y comportamiento */
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");

    /* ¡CORRECCIÓN CRÍTICA AQUÍ! Sin comillas, es un long, no un string */
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); 

    /* Límite de descarga opcional */
    if (max_bytes > 0) {
        curl_easy_setopt(curl, CURLOPT_MAXFILESIZE_LARGE, (curl_off_t)max_bytes);
    }

    /* 4. Cabeceras HTTP idénticas a las de un navegador real */
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8");
    headers = curl_slist_append(headers, "Accept-Language: es-ES,es;q=0.9,en;q=0.8");
    headers = curl_slist_append(headers, "Connection: keep-alive");
    headers = curl_slist_append(headers, "Upgrade-Insecure-Requests: 1");
    headers = curl_slist_append(headers, "Sec-Fetch-Dest: document");
    headers = curl_slist_append(headers, "Sec-Fetch-Mode: navigate");
    headers = curl_slist_append(headers, "Sec-Fetch-Site: none");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    /* 5. Ejecutar la petición */
    CURLcode res = curl_easy_perform(curl);

    /* 6. Obtener código HTTP ANTES de destruir el handle */
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    /* 7. Limpieza de recursos de libcurl (UNA SOLA VEZ, al final de la configuración) */
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    /* ---- Manejo de errores ---- */
    if (res != CURLE_OK) {
        fprintf(stderr, "peticionURL: error de red en '%s': %s\n", url, curl_easy_strerror(res));
        free(buf.data);
        return NULL;
    }

    if (http_code < 200 || http_code >= 300) {
        fprintf(stderr, "peticionURL: HTTP %ld en '%s'\n", http_code, url);
        free(buf.data);
        return NULL;
    }

    if (buf.size == 0) {
        fprintf(stderr, "peticionURL: respuesta vacía en '%s'\n", url);
        free(buf.data);
        return NULL;
    }

    /* Éxito: devolvemos los datos. El llamador es responsable de free() */
    return buf.data;
}

/* ------------------------------------------------------------------
 * Realiza la llamada al buscador DuckDuckGO para recoger enlaces
 *          https://api.duckduckgo.com/?q=clave&format=json&no_html=1"
 *          cada enlace se marca FUNC(url:<enlace>)\n
 *
 * p   		claves a buscar en el buscador
 * resultado   	Dara los enlaces de la busqueda con el formato FUNC
 * maxResultado	Maximo tamaño del resultado
 *
 * ------------------------------------------------------------------ */
void cmdBusca(const char *p, char *resultado, int maxResultado) 
{ 
   printf("Realizamos la busqueda de la consulta: %s\n", p); 
   // Lanzamos URL con busqueda

   // Escapamos la consulta y Creamos URL
   CURL *curl = curl_easy_init();
   char *preguntaEscapada = curl_easy_escape(curl, p, 0);
   char url[1024];  // URL, no sobrepasara los 1024
   snprintf(url, sizeof(url), "https://api.duckduckgo.com/?q=%s&format=json&no_html=1", preguntaEscapada);
   curl_free(preguntaEscapada);

   printf("Consulta Duck %s\n",url);
   char *html = peticionURL(url, 1024 * 1024); // 1MB de JSON
   if (html) 
   {
       // Con el JSON, queda recoger las URLs
       obtenerURLsJSON(html, resultado, maxResultado);
       printf("-----------------------\n%s--------------------\n",resultado);
       free(html);
   } 
   else { fprintf(stderr, "No se pudo obtener la página.\n"); }

   // ¡IMPORTANTE! Limpiar el handle al terminar de usarlo
   curl_easy_cleanup(curl);
   curl = NULL;

}

/* ------------------------------------------------------------------
 * Recurivamente extrae el texto eliminando <scripts <style>
 *
 * Parametros: node:  nodo,  resultado, maxResultado
 * ------------------------------------------------------------------ */
static void extraerTextoNodo(xmlNode *node, char *resultado, size_t maxResultado) 
{
   for (xmlNode *cur = node; cur; cur = cur->next) 
   {
      if (cur->type == XML_TEXT_NODE) // Nodo texto
      {
         if (cur->content) 
         {
            // Añadir el texto del nodo al resultado
            strncat(resultado, (char *)cur->content, maxResultado - strlen(resultado) - 1);
         }
      } 
      else if (cur->type == XML_ELEMENT_NODE) // Nodo elemento
      {
         // Ignorar completamente los bloques:
         if ( strcasecmp((char *)cur->name, "head") != 0 && strcasecmp((char *)cur->name, "script") != 0 &&
              strcasecmp((char *)cur->name, "meta") != 0 && strcasecmp((char *)cur->name, "link") != 0 &&
              strcasecmp((char *)cur->name, "style") != 0 && strcasecmp((char *)cur->name, "footer") != 0 &&
              strcasecmp((char *)cur->name, "noscript") != 0 && strcasecmp((char *)cur->name, "header") != 0 &&
              strcasecmp((char *)cur->name, "iframe") != 0 && strcasecmp((char *)cur->name, "object") != 0 &&
              strcasecmp((char *)cur->name, "embed") != 0 && strcasecmp((char *)cur->name, "svg") != 0 &&
              strcasecmp((char *)cur->name, "aside") != 0 && strcasecmp((char *)cur->name, "nav") != 0 )
         {
            extraerTextoNodo(cur->children, resultado, maxResultado);
         }
      }
   }
}

/* ------------------------------------------------------------------
 * Del texto HTML deja solo texto leible
 *
 * Parametros:
 * 	html_str   Texto en formato HTML
 * 	resultado
 * 	 maxResultado
 * ------------------------------------------------------------------ */
static void limpiaHTML(const char *html_str, char *resultado, size_t maxResultado) 
{
   resultado[0] = '\0';   // Inicialmente a vacio

   //  Parsear HTML a arbol DIM en memoria (NULL para codificación automática)
   htmlDocPtr doc = htmlReadMemory(html_str, (int)strlen(html_str), "", NULL, 
                          HTML_PARSE_NOBLANKS | HTML_PARSE_NONET | HTML_PARSE_NOWARNING | HTML_PARSE_NOERROR);
   
   if (doc == NULL) { fprintf(stderr, "Error al parsear el HTML\n"); return; }

   // Obtener el nodo raíz del documento
   xmlNode *root_element = xmlDocGetRootElement(doc);

   //  Extraer el texto recursivamente
   if (root_element) { extraerTextoNodo(root_element, resultado, maxResultado); }

   // Liberar los recursos de libxml2
   xmlFreeDoc(doc);
   xmlCleanupParser();
}

/**
 *  Limpia una cadena de texto colapsando todos los espacios en blanco consecutivos (espacios, 
 *  tabs, saltos de línea) en un único espacio.
 *  También elimina espacios al inicio y al final.
 * 
 * Parametros:
 * 	entrada La cadena de texto original (no se modifica).
 * 	salida  Resultado sin los espacio tab, etc.
 * Retorno: 0 correcto negativo error
 */
int limpiaEspacios(const char *entrada, char *salida) 
{
   if (entrada == NULL) { return -1; }

   size_t longitud = strlen(entrada);
   // Asignamos memoria máxima posible (igual que la entrada) + 1 para el '\0'
   //char* salida = (char*)malloc(longitud + 1);
   //if (salida == NULL) { fprintf(stderr, "Error crítico: Fallo en malloc para limpiar texto.\n"); return NULL; }

   size_t j = 0;          // Índice para la cadena de salida
   int en_espacio = 1;    // Estado: 1 si el último carácter procesado fue espacio (para ignorar espacios iniciales)

   for (size_t i = 0; i < longitud; i++) 
   {
      // Usamos (unsigned char) para evitar comportamiento indefinido con caracteres negativos en ctype.h
      if (isspace((unsigned char)entrada[i])) 
      {
         if (!en_espacio) 
         {
            salida[j++] = ' '; // Colapsamos cualquier secuencia de whitespace a un solo espacio
            en_espacio = 1;
         }
       } 
       else 
       {
          salida[j++] = entrada[i];
          en_espacio = 0;
       }
   }

   // Si la cadena original terminaba en espacios, 'j' apuntará a un espacio final. Lo eliminamos.
   if (j > 0 && salida[j - 1] == ' ') { j--; }

   salida[j] = '\0'; // Terminador nulo

   // Optimización de memoria: reducimos el bloque al tamaño exacto necesario
   //char* resultado_optimizado = (char*)realloc(salida, j + 1);
   // Si realloc falla, es seguro seguir usando el puntero original 'salida'
   //return (resultado_optimizado != NULL) ? resultado_optimizado : salida;

   return 0;
}

/* ------------------------------------------------------------------
 * Solicita una pagina HTML y extrae texto plano
 *
 * Parametros:
 * 	url    URL de la pagina
 * 	resultado   Dara el texto en plano en formato FUNC(reduce:<texto> )\n"
 * 	maxResultado    Maximo tamaño del resultado
 *
 * ------------------------------------------------------------------ */
void cmdURL(const char *url, char *resultado, int maxResultado) 
{ 
  resultado[0] = '\0' ;   // Vaciamos resultado
  //printf("Recupera Información de la URL %s y la pasa a texto plano\n", url); 

  if (!url || !*url) { fprintf(stderr, "cmdUrl: URL vacía o NULL.\n"); return ; }

  printf("Obtenemos información HTM: %s\n",url);
  char *html = peticionURL(url, 10 * 1024 * 1024); // 2MB maximo de HTML
  if (html)
  {
       printf("\tTamaño descargado en HTML: %zu bytes de url %s ", strlen(html),url);
       // Limpiamos Datos.
       char resulLimpio[10 * 1024];
       limpiaHTML(html,resulLimpio,sizeof(resulLimpio));
       free(html); // Ya no necesitamos el HTML crudo 
       limpiaEspacios(resulLimpio, resultado);
       //snprintf(resultado, maxResultado, "%s\n", resulLimpio);
       printf("\tLIMPIO %zu\n",strlen(resulLimpio));
  }
  else { fprintf(stderr, "No se pudo obtener la página.\n"); }
}
