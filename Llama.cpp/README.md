# proyectos y ejemplos IA
  
   Ejemplos y proyectos con API Llama.cpp programando en C  
  
    Antes de nada descargar e instalar el entorno llama.cpp, en este caso se guarda en directorio llama.cpp antes de la estructura de directorio
  
    -  T- \llama.cpp      Directorio de API llama.cpp
    -  |----build            Compilación para Linux
    -  |----buildWindows     Compilación para Windows
    -  |- \bin            Directorio con ejecutables compilado
    -  |- \modelos        Donde normalmente descargaremos los modelos
    -  |- \roles          Ficheros con Roles según aplicaciones.
    -  |- \src            Fuentes
    -  |----\includes        Includes comunes que no son del llama.cpp
    -  |----\<Targets>          Los ejemplos y proyectos, Simple, Base......
    -  | MakeFile         Objetivos de compilación para la compilación de los ejemplos
<BR>
Construcción de librerias llama.cpp en estatico, para llevar los ejemplos con facilidad a diversos entornos.  
 > git clone https://github.com/ggml-org/llama.cpp  
 > cd llama.cpp  
  
 # Linux  
 > cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64ilp   
 > cmake --build build --config Release -j 
     
 # Windows  
 > cmake -B build-windows -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ -DBUILD_SHARED_LIBS=OFF  
 > cmake --build build-windows --config Release -j  
<BR>
#Ejemplos y proyectos  
  
  - **[Base](src/Base)**  Ejemplo basico para crear el primer proyecto.  
  - **[Agente](src/Agente)**  Ejemplo de realización de una agente para preguntarle la hora ye el clima, programado para almacenar contexto.  

<br><br>
__Website__: <https://www.pinguytaz.net>

