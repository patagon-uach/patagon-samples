# Clasificacion de Imagenes utilizando Pytorch en el supercomputador Patagon.

## Introduccion

En este pequeno tutorial repasaremos lo necesario para realizar tareas de un tipico pipeline **Machine Learning** utilizando **Python**. El objetivo es hacer mas asequible el uso del supercomputador **Patagon** y (ojala!) despejar dudas que puedan haber quedado luego de repasar el manual de usuario al ver un ejemplo concreto.
Este tutorial hara uso de la informacion entregada en las distintas entradas al manual de usuario del supercomputador **Patagon de la UACh**, las cuales estan disponibles en [patagon.uach.cl](https://patagon.uach.cl).

La libreria a utilizar en este caso en particular es **Pytorch**, sin embargo, el proceso se puede adaptar para utilizar cualquier otra libreria.

Una vista general del tutorial sera:

1. Administracion del contenedor **Docker**
2. Instalacion de librerias y paquetes necesarios
3. Ejecucion de la tarea utilizando **SLURM**

Como requisito a este tutorial se requiere de acceso al **Supercomputador Patagon** y haber clonado el repositorio de muestras **Patagon-Samples** disponible en [**Github**.](https://github.com/patagon-uach/patagon-samples)
Para ello basta con ejecutar:

```
$ git clone https://github.com/patagon-uach/patagon-samples
```

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

```
srun --container-name=python3.8 --container-image='python:3.8' -p cpu --pty bash
```

Aca **container-name** es un nombre arbitrario para identificar al contenedor de los demas que hayas descargado y **container-image** es el nombre del contenedor de **Docker Hub**. En este caso especificamos la version de Python que requiero dentro del contenedor utilizando ':'. Se especifica la particion `cpu` para no reservar ninguna GPU y alocar la minima cantidad de recursos, pues no es una tarea importante.

En respuesta al comando anterior veremos como se descarga el contenedor:

```
pyxis: importing docker image ...
```

Una vez terminado, estaremos dentro del contenedor en una sesion `bash`:

```
user@nodeGPU01:~$
```

Es importante destacar que cada vez que se ejecute el comando `srun` en el **nodo maestro** la instruccion que le sigue se llevara a cabo dentro del **nodo de computo**, en el comando anterior, al estar ejecutando `bash` se estan reservando **recursos** del **nodo de computo** y abriendo una sesion **interactiva**. Es de suma importancia mantener el ingreso **interactivo** al **nodo de computo** al minimo para no desperdiciar los limitados recursos y utilizarlo solo para instalar paquetes. \
Luego podemos ingresar `exit` para salir, lo que nos hara volver al **nodo maestro** y desocupar los recursos que reservamos. Sin embargo, por ahora nos quedaremos un ratito mas para instalar la libreria que necesitamos en nuestro contenedor.
Afortunadamente, utilizando `pip` uno puede instalar liberias a nivel de usuario sin necesidad de privilegios root. Instalaremos **PyTorch** usando:

```
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --user
```

\*: comando de instalacion obtenido de [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Cuando finalice la instalacion podemos ingresar a una sesion interactiva de **Python** usando `python` y probar con `import torch`.

```
ttest@nodeGPU01:~$ python
Python 3.8.12 (default, Mar  2 2022, 04:56:27)
[GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>>
```

0 errores!.

En este punto estariamos dentro de una sesion `ssh` en el nodo maestro, la cual mantiene una sesion interactiva dentro del nodo de computo, dentro del contenedor la que esta ejecutando python interactivamente (inception!).

Si se ha instalado correctamente podemos salir de la sesion interactiva de **Python** con `CTRL+D` o ingresando `exit()`.

Finalizamos la sesion interactiva `bash` con `exit`.

Una vez de vuelta en el **nodo maestro**, podemos volvemos a ingresar al contenedor pero esta vez utilizando el flag --container-remap-root

```
srun --container-name=python3.8 --container-image='python:3.8' -p cpu --container-remap-root --pty bash
```

de esta manera ingresaremos al contenedor como root, lo que nos permitira instalar paquetes a nivel de sistema en nuestro ambiente. En este caso particular, el ambiente que viene con el contenedor docker esta basado en ubuntu, por lo que el gestor de paquetes es `apt`. Para instalar un paquete bastaria con ejecutar:

```
apt update
apt install <nombre del paquete>
```

Al momento de escribir este tutorial, se es consciente de que muchos proyectos de investifacion relacionados con machine learning utilizan conda como gestor de librerias y ambientes virtuales en python. Sin embargo, todavia se esta explorando como utilizar ambientes virtuales conda en conjunto con los contenedores.
Afortunadamente, como el patagon es basdo en contenedores, uno ya tiene su ambiente virtual con acceso root, lo que podria servir para instalar librerias que se requieran a nivel de sistema, una de las caracteristicas mas utiles de conda\*.

## 3. Ejecucion de la tarea utilizando **SLURM**

De vuelta en el **nodo maestro** con nuestro contenedor preparado para cualquier tarea que requiera **Pytorch**, procederemos a crear un script de **SLURM**, con toda la informacion necesaria para la ejecucion de nuestra tarea.

**MiTareaML.slurm**

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
cd patagon-samples/05-Tutoriales                    # navegacion al
cd 01-machine-learning-pytorch                      # directorio
srun --container-name=python3.8 python main.py
```

Para mayor informacion sobre el script y el funcionamiento de **SLURM** ver [aca](https://patagon.uach.cl/patagon/tutoriales/how-to-launch-slurm-jobs.html).

Por ahora dejaremos este **MiTareaML.slurm** en el directorio raiz de nuestro usuario. \
Finalmente ejecutaremos

```
$ sbatch MiTareaML.slurm
Submitted batch job 20471
```

para manadr nuestro **job** a la cola de **SLURM**.

Haciendo `cat MiTareaML-<jobid>.out` podemos monitorear la salida de nuestro **job**:

```
$ cat MiTareaML-20471.out
/home/user
Sat 12 Mar 2022 10:49:32 PM -03
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.329474
Train Epoch: 1 [640/60000 (1%)]	Loss: 1.425025
Train Epoch: 1 [1280/60000 (2%)]	Loss: 0.797880
```

y con `squeue` el estado en la cola de **SLURM**.

```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             20195       gpu  python3 useruser  R 6-11:02:57      1 nodeGPU01
             20299       gpu importan    user2  R 3-09:53:03      1 nodeGPU01
             20471       gpu MiTareaM     user  R       0:04      1 nodeGPU01
```

Podemos usar `scancel` para cancelar la ejecucion de nuestro **job**:

```
$ scancel 20471
```

Si somos afortunad@s lo veremos en ejecucion inmediatamente. En caso contrario, se encolara hasta que hayan recursos disponibles para su ejecucion. \
Ahora podemos cerrar tranquilamente la sesion `ssh` sin miedo a que se cancele nuestro **job**, **SLURM** se encargara del resto.

Con esto damos por concluido el primer tutorial sobre el uso del **Supercomputador Patagon de la UACh**, gracias por leer y les deseamos mucho exito en sus investigaciones.

Cualquier comentario sobre este documento sera bien recibido. Lo pueden hacer llegar al correo del patagon patagon@uach.cl.
