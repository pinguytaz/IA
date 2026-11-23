# GestorLlamaCPP
  
   Proyecto par poder cargar varios modelos (-GGUF) en memoria y ejecutar inferencia con el que deseemos, definiendo la configuración que nos interese en cada caso.  
   Permite ser un pequeño marco de trabajo en el que solo debemos preocuparnos de nuestra programación, olvidando las cargas e inferencias de los modelos, y solo debemos aportar los paramentros de configuración al modelo como son los roles, sampler y parametros de contexto si no estamos conformes con los de por defecto.  


<div style="background-color: black; color: #00FF00; padding: 20px; font-family: monospace; border-radius: 5px;">
<pre>
                                    ________________________________________________________________
                                   / [ IA-LLAMA.CPP ] _____________________________________________ \
                                   |                                                                |
                                   |   _________________________________________________________    |
                                   |  /                                                         \   |
                                   |  |  [SYSTEM READY]                                         |   |
                                   |  |  > LOADING Modelo GGUF. [OK]                            |   |
                                   |  |  > LOADING Roles....... [OK]                            |   |
                                   |  |  > LOADING Confg....... [OK]                            |   |
                                   |  |  > REGEX ENGINE....... [POSIX]                          |   |
                                   |  |                                                         |   |
                                   |  |  > DETECTING FUNC(CALL:EXTRACT_LOGIC)...                |   |
                                   |  \_________________________________________________________/   |
                                   |                                                                |
                                   |            [ 128K KV-CACHE ] [ GCC-MINGW64 ]                   |
                                   \________________________________________________________________/
                                            ||                                           ||
                                       ____/  \_________________________________________/  \____
                                      /                                                         \
</pre>
</div>
  
    Antes de nada descargar e instalar el entorno llama.cpp, en este caso se guarda en directorio llama.cpp antes de la estructura de directorio
  
    -  T- ..\llama.cpp      Directorio de API llama.cpp  
    -  |------build         Compilación para Linux  
    -  |----..buildWindows  Compilación para Windows  
    -  |- ..\modelos        Donde normalmente descargaremos los modelos  
    -  |- ..\roles          Ficheros con Roles según aplicaciones.  
    -  |- \bin              Directorio con ejecutables compilados  
    -  |- \src              Fuentes del GestorLlamaCPP  
    -  |- \auxiliares       Funciones auxiliares que pueden servir den diversos proyectos  
    -  |- \ejemplos         Ejemplos de uso del GestorLlamaCPP  
    -  | MakeFile         Objetivos de compilación para la compilación de los ejemplos
<BR> <BR>
# Ejemplos y proyectos  
  
  - **[EjemploUso](src)**  Ejemplo basico de uso del gestor.  
  - **[LaBusquedaBase](src/LaBusqueda)**  Ejemplo, con 4 modelos en memoria, se le pasa una pregunta y realiza las siguienetes acciones:
-  - LLM1: Localiza las consultas a realizar a los buscadores,  
            Con las busquedas detectadas por LLM, se realiza consultas a buscador y recupera la información de las URLs aportadas
-  - LLM2: Resumen de todas las paginas descargadas.  
-  - LLM3: Generación del informe con los resumenes realizados (2 informes 2 modelos diferentes)  
-  - LLM4 Analisi de los informes generados y elección de uno.  
 
# Uso  
  
  Lo primero es inicializar el gestor con **iniciaGestor(false)**   
  NO olvidar limpiarlo antes de salir con **destructirGestor()** 
  
  Ahora nos queda cargar los modelos que deseemos con **cargaModelo("Un Nombre", "Nombre Fichero")** le daremos un nombre, ya sea el propio del modelo o algo que nos de información de lo que realizara, esta funcion nos dara un número entero que se debera guardar ya que es la referencia del modelo y es con el que accdderemos a funciones de inferencia o recogida de información como el nombre del archivo o el nombre puesto.  
  
  *Funciones del Gestor:*  
  
  -   **void iniciaGestor(bool log)** Inicializamos el motor, iniciando el backend.  
  -   **void destructorGestor(void)** Limpia el gestor de todos los modelos y del backend.  
  -   **void limpiaGestor(void)** Elimina modelos del gestor pero continua backend.  
  -   **int cargaModelo(const char *nombre, char* pathModelo)** Cargamos un modelo en memoria, retornara su identificador.  
  -   **void listaModelos(void)** Lista los modelos cargados, dandonos el nombre puesto y su identificador.  
  -   **int modelosCargados(void)** Numero de modelos cargado  
  
<BR>
  
  *Funciones sobre modelos cargados:*  
  
  -   **int borraModelo(int id)** Borra el modelo indicado.  
  -   **char *get_NombreModelo(int indice)** Nombre puesto al modelo;  
  -   **char *get_NombreArchivo(int indice)** Nombre del archivo del modelo  
  -   **int32_t vocab_n_tokens(int id)**   Nos el total de tokens del vocabulario par Sampler
  -   **paramContext_t get_ParamContexto(int id)** Parametros definidos del contexto del modelo.  
  -   **int put_ParamContexto(int id, paramContext_t parametros)**  cambiamos los parametros del modelo del contexto.  
  -   **int inferir( ..PARAMETROS.. )**  Realiza la inferencia y retorna 0 si todo OK.  
  -   -  int   Identificador del modelo  
  -   -  const char* Pregunta.  
  -   -  char * Rol del sistema, si no queremos ninguno le pondremos ""  
  -   -  sampler_t cadena de sampler, parametros de temperatur, k, p, etc.  
  -   -  char * respuesta.  respuesta, 
  -   -  int Tamaño de maximo de la respuesta
  -    - bool indicamos si queremos limpiar el contexto antes de iniciar la inferencia, o si queremos mantener la información de la inferencia pasada.
  -    - int Maximo número de tokens que la inferencia buscar para darnos respuesta.
  -    - LimpiaContexto_t    Estrategia de limpieza del contexto:  VENTANA_DESLIZANTE, COMPLETO
  
<BR>

<br><br>
__Website__: <https://www.pinguytaz.net>

