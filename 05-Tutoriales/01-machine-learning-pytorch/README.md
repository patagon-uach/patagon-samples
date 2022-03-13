# Clasificacion de Imagenes utilizando Pytorch en el supercomputador Patagon.

## Introduccion

En este pequeno tutorial repasaremos lo necesario para realizar tareas de un tipico pipeline **Machine Learning** utilizando **Python**. El objetivo es hacer mas asequible el uso del supercomputador Patagon y (ojala!) despejar dudas que puedan haber quedado luego de repasar el manual de usuario al ver un ejemplo concreto.
Este tutorial hara uso de la informacion entregada en las distintas entradas al manual de usuario del supercomputador Patagon de la UACh, las cuales estan disponibles en [patagon.uach.cl](https://patagon.uach.cl).

La libreria a utilizar en este caso en particular es Pytorch, sin embargo, el proceso se puede adaptar para utilizar cualquier otra libreria.

Una vista general del tutorial sera:

1. Administracion del contenedor Docker
2. Instalacion de librerias y paquetes necesarios
3. Ejecucion de la tarea utilizando **SLURM**

## 1. Administracion del contenedor Docker

El supercomputador Patagon consiste principalmente de 2 nodos de procesamiento conocidos como **Patagon-Master** (**nodo maestro**) y **nodeGPU01** (**nodo de computo**). Cada uno tiene un objetivo distinto y es de suma importancia respetar el uso que deben tener.

El **nodo maestro** es al cual uno ingresa al supercomputador cuando se conecta por `ssh`. Este nodo solo tiene un objetivo para el usuario: recibir trabajados y transmitirlos al nodo de computo para su ejecucion. \
Es de suma importancia no ejecutar tareas que demanden una alta capacidad de computo en el nodo maestro (nada de `./simulacionUltraCostosa` ni `./entrenarMiRed.py`), lo ideal es solo ejecutar tareas basicas para el manejo de directorios/archivos (`cat`, `mkdir`, `ls`, `cd`, `vim`, `nano`, etc...) y para lanzar trabajos al nodo de computo utilizando **SLURM**. Para mayor infromacion sobre la arquitectura del Patagon ver [aqui](https://patagon.uach.cl/patagon/especificaciones-tecnicas.html).

El **nodo de computo** contiene la principal capacidad de computo del patagon LINK, por lo que todas nuestras tareas de investigacion deberian ejecutarse ahi, mandadas desde el **nodo maestro**. \
Novedosamente, el **nodo de computo** funciona en base a contenedores **Docker** utilizando las herramientas **Pyxis** y **Enroot**, lo que nos permite mantener multiples ambientes **privados** por usuario sin la necesidad de instalar librerias a nivel de sistema. Con esta forma de funcionamiento, cada usuario es responsable de administrar su ambiente, esto incluye instalar librerias y programas que pueda necesitar. Para mas informacion revisar [aca](https://patagon.uach.cl/patagon/tutoriales/administracion-contenedores.html).

## 2. Instalacion de librerias y paquetes necesarios

Para descargar un contenerdor podemos ingresar a [https://hub.docker.com](https://hub.docker.com)
donde se encuentran miles de contenedores con distintos paquetes pre-instalados, lo cual nos servira como un punto de partida. \
En nuestro caso, dado que nos interesa utilizar **Python** con la libreria **Pytorch** podriamos buscar un contenedor que ya tenga **Pytorch** instalado, sin embargo, para efectos practicos utilizaremos un contenedor de python estandar, para hacernos la vida mas dificil:

[https://hub.docker.com/\_/python](https://hub.docker.com/_/python)

**Importante: Es posible que para descargar el contenedor que necesites debas seguir los pasos de [esta entrada al manual](https://patagon.uach.cl/patagon/tutoriales/autentificacion-contenedores.html).**

`srun --container-name=python3.8 --container-image='python:3.8' -p cpu --pty bash`

Aca **container-name** es un nombre arbitrario para identificar al contenedor de los demas que hayas descargado y **container-image** es el nombre del contenedor de **Docker Hub**. En este caso especificamos la version de Python que requiero dentro del contenedor utilizando ':'. Se especifica la particion `cpu` para no reservar ninguna GPU y alocar la minima cantidad de recursos, pues no es una tarea importante.

Es importante destacar que cada vez que se ejecute el comando `srun` en el **nodo maestro** la instruccion que le sigue se llevara a cabo dentro del **nodo de computo**, en el comando anterior, al estar ejecutando `bash` se estan reservando **recursos** del **nodo de computo** y abriendo una sesion **interactiva**. Es de suma importancia mantener el ingreso **interactivo** al **nodo de computo** al minimo para no desperdiciar los limitados recursos y utilizarlo solo para instalar paquetes. \
Luego podemos ingresar `exit` para salir, lo que nos hara volver al **nodo maestro** y desocupar los recursos que reservamos. Sin embargo, por ahora nos quedaremos un ratito mas para instalar la libreria que necesitamos en nuestro contenedor.
Afortunadamente, utilizando `pip` uno puede instalar liberias a nivel de usuario sin necesidad de privilegios root. Instalaremos **PyTorch** usando:

`python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --user`

\*: comando de instalacion obtenido de [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Cuando finalice la instalacion podemos ingresar a una sesion interactiva de **Python** usando `python` y probar con:

`import torch`

En este punto estariamos dentro de una sesion `ssh` en el nodo maestro, la cual mantiene una sesion interactiva dentro del nodo de computo, dentro del contenedor la que esta ejecutando python interactivamente (inception!).

Si se ha instalado correctamente podemos salir de la sesion interactiva de **Python** con `CTRL+D` o ingresando `exit()`.

Procedemos a instalar los paquetes restantes usando:

`python -m pip install <> --user`

Finalizamos con `exit`.

Una vez de vuelta en el **nodo maestro**, podemos volvemos a ingresar al contenedor pero esta vez utilizando el flag --container-remap-root

`srun --container-name=python3.8 --container-image='python:3.8' -p cpu **--container-remap-root** --pty bash`

de esta manera ingresaremos al contenedor como root, lo que nos permitira instalar paquetes a nivel de sistema en nuestro ambiente. En este caso particular, el ambiente que viene con el contenedor docker esta basado en ubuntu, por lo que el gestor de paquetes es `apt`. Para instalar un paquete bastaria con ejecutar:
`apt update`
`apt install libglm-dev`

Al momento de escribir este tutorial, se es consciente de que muchos proyectos de investifacion relacionados con machine learning utilizan conda como gestor de librerias y ambientes virtuales en python. Sim embargo, conda exige la utilizacion de ambientes virtuales para instalar librerias, lo que.
Afortunadamente, como el patagon es basdo en contenedores, uno ya tiene su ambiente virtual con acceso root, lo que podria servir para instalar librerias que se requieran a nivel de sistema, una de las caracteristicas mas utiles de conda\*.

## 3. Ejecucion de la tarea utilizando **SLURM**

De vuelta en el **nodo maestro** con nuestro contenedor preparado para cualquier tarea que requiera pytorch, procederemos a crear un script de slurm, con toda la informacion que se necesita para la ejecucion de nuestra tarea.

MiTareaML.slurm

```
#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p gpu                       # Particion GPU
#SBATCH --gres=gpu:A100:1            # Una GPU por favor

# OTHER PARAMS
#SBATCH -J MiTareaML          # Nombre de la tarea
#SBATCH -o MiTareaML-%j.out   #
#SBATCH -e MiTareaML-%j.err   #

# COMMANDS ON THE COMPUTE NODE
pwd                         #
date                        #

# Ejecutar nuestro programa
srun --container-name=python3.8 python <>
```

Para mayor informacion sobre el script y el funcionamiento de slurm ver LINK.

Haciendo `cat MiTareaML-<jobid>.out` podemos monitorear la salida. y con `squeue` el estado en la cola de SLURM. Si somos afortunad@s lo veremos en ejecucion inmediatamente. En caso contrario la ejecucion se encolara hasta que hayan recursos disponibles. Ahora podemos cerrar tranquilamente la sesion ssh sin miedo a que se cancele el job, slurm se encargara del resto.
Con esto damos por concluido el primer tutorial sobre el uso del patagon, gracias por leer y les deseamos mucho exito en sus investigaciones.

Cualquier comentario sobre este documento sera bien recibido. Lo pueden hacer llegar al correo del patagon patagon@uach.cl.
