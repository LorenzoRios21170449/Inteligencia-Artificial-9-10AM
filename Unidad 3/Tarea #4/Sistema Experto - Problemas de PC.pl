%--------------------------------------------------------------------
% SISTEMA EXPERTO SIMPLE DE DIAGNOSTICO DE PC
% Base de Conocimiento y Motor de Inferencia
%--------------------------------------------------------------------
:- dynamic(sintoma_ingresado/1).

% Predicado para reiniciar el estado del sistema (limpiar hechos dinamicos)
reiniciar :-
    retractall(sintoma_ingresado(_)).

% Muestra una sugerencia al usuario
sugerir(Texto) :-
    writeln(''),
    writeln(['Sugerencia: ', Texto]),
    writeln('').

% Muestra una conclusion o diagnostico
concluir(Texto) :-
    writeln(''),
    writeln(['CONCLUSION: ', Texto]),
    writeln('').

% Muestra un aviso/notificacion importante
notificar(Texto) :-
    writeln(''),
    writeln(['AVISO: ', Texto]),
    writeln('').

% Convierte las cadenas a minusculas para una comparacion sin distinguir mayusculas/minusculas.
contiene_sintoma(TextoUsuario, FraseClave) :-
    string_lower(TextoUsuario, TextoMinusculas),
    string_lower(FraseClave, FraseMinusculas),
    sub_string(TextoMinusculas, _, _, _, FraseMinusculas).

entrada_usuario(String) :-
    read_line_to_string(user_input, String).

%------------------------------------------------------------------
% Motor de Inferencia / Flujo Principal del Sistema Experto
%--------------------------------------------------------------------

% Punto de entrada al sistema experto
iniciar_diagnostico :-
    reiniciar, 
    writeln(''),
    writeln('--- Asistente tecnico en diagnosticos de problemas de computadora ---'),
    writeln('Describe el problema que estas experimentando.'),
    writeln('Escribe tu descripcion y presiona Enter.'),
    % Se lee la entrada del usuario 
    entrada_usuario(Descripcion),
    % Aserta el sintoma ingresado como un hecho dinamico
    assertz(sintoma_ingresado(Descripcion)),
    writeln('Analizando...'),
    % Intenta diagnosticar basado en el sintoma ingresado
    diagnosticar_problema,
    % Limpia el sintoma ingresado al finalizar el diagnostico inicial
    retractall(sintoma_ingresado(_)).

% Predicado que intenta aplicar las reglas de diagnostico basado en el sintoma ingresado
diagnosticar_problema :-
    sintoma_ingresado(Descripcion),
    (   diagnosticar(Descripcion, _)
    ->  true
    ;   (   contiene_sintoma(Descripcion, "error")
        ;   contiene_sintoma(Descripcion, "problema")
        ;   contiene_sintoma(Descripcion, "no anda")
        )
    ->  writeln("Podrías describir el problema con más detalle? ¿Qué está ocurriendo exactamente?"),
        write("Descripción: "),
        read_line_to_string(user_input, NuevaDescripcion),
        retractall(sintoma_ingresado(_)),          % Borra el sintoma vago anterior
        assertz(sintoma_ingresado(NuevaDescripcion)), % Inserta el nuevo
        diagnosticar_problema                      
    ;   diagnostico_por_defecto
    ).



%--------------------------------------------------------------------
% Base de Conocimiento: Reglas de Diagnostico
% Estructura: diagnosticar(DescripcionUsuario, NombreInternoDelProblema) :- Condicion, Acciones.
% La Condicion usa 'contiene_sintoma' y ';' para OR.
% Las Acciones usan ',', 'sugerir', 'concluir', 'notificar'.
%--------------------------------------------------------------------

% Regla: PC lenta
diagnosticar(Descripcion, pc_lenta) :-
    (   contiene_sintoma(Descripcion, 'PC lenta');
        contiene_sintoma(Descripcion, 'computadora tarda');
        contiene_sintoma(Descripcion, 'se pasma mucho')
    ),
    sugerir('Eliminar programas innecesarios desde ''Agregar o quitar programas''.'),
    sugerir('Desactivar aplicaciones que se inician automaticamente con Windows (desde el Administrador de Tareas).'),
    sugerir('Liberar espacio en disco ejecutando la herramienta ''Limpieza de disco'' de Windows.').

%Regla: Conexión Wi-Fi inestable
diagnosticar(Descripcion, wifi_inestable) :-
    (   contiene_sintoma(Descripcion, 'Wi-Fi inestable');
        contiene_sintoma(Descripcion, 'internet se desconecta');
        contiene_sintoma(Descripcion, 'señal debil de Wi-Fi')
    ),
    sugerir('Reiniciar el router y el modem (desconectarlos de la corriente por 30 segundos).'),
    sugerir('Verificar que los cables del router/modem esten bien conectados.'),
    sugerir('Acercar la PC al router o eliminar obstaculos.').

% Regla: Anuncios emergentes
diagnosticar(Descripcion, anuncios_emergentes) :-
    (   contiene_sintoma(Descripcion, 'anuncios emergentes');
        contiene_sintoma(Descripcion, 'pop-ups de publicidad');
        contiene_sintoma(Descripcion, 'navegador con mucha publicidad')
    ),
    sugerir('Descargar y ejecutar un escaneo completo con un software antivirus actualizado.'),
    sugerir('Revisar la lista de programas instalados y eliminar cualquier software que no reconozca (desde ''Agregar o quitar programas'').'),
    sugerir('Restablecer la configuracion del navegador web a sus valores predeterminados.').

% Regla: Periféricos no responden
diagnosticar(Descripcion, perifericos_no_responden) :-
    (   contiene_sintoma(Descripcion, 'mouse no funciona');
        contiene_sintoma(Descripcion, 'teclado no escribe');
        contiene_sintoma(Descripcion, 'periferico USB no detecta');
        contiene_sintoma(Descripcion, 'dispositivo USB no responde')
    ),
    sugerir('Verificar que el periferico este bien conectado a la PC.'),
    sugerir('Probar el periferico en otro puerto USB diferente.'),
    sugerir('Reiniciar la computadora.'),
    sugerir('Si es inalambrico, verificar las baterias o la conexion del receptor.').

% Regla: Espacio en disco lleno
diagnosticar(Descripcion, espacio_disco_lleno) :-
    (   contiene_sintoma(Descripcion, 'espacio en disco lleno');
        contiene_sintoma(Descripcion, 'queda poco espacio');
        contiene_sintoma(Descripcion, 'disco C esta rojo')
    ),
    sugerir('Ejecutar la herramienta ''Limpieza de disco'' para eliminar archivos temporales.'),
    sugerir('Revisar la carpeta de Descargas y eliminar archivos grandes que no necesite.'),
    sugerir('Considerar mover archivos grandes (fotos, videos, documentos) a un disco duro externo o servicio en la nube.'),
    sugerir('Desinstalar programas grandes que no utilice.').

% Regla: Errores de fecha y hora
diagnosticar(Descripcion, fecha_hora_mal) :-
    (   contiene_sintoma(Descripcion, 'fecha y hora mal');
        contiene_sintoma(Descripcion, 'hora incorrecta');
        contiene_sintoma(Descripcion, 'problema con la fecha')
    ),
    sugerir('Hacer clic en el reloj en la barra de tareas y seleccionar ''Ajustar fecha y hora''.'),
    sugerir('Asegurarse de que la opcion ''Establecer hora automaticamente'' este activada.'),
    sugerir('Asegurarse de que la opcion ''Establecer zona horaria automaticamente'' este activada o seleccionar la zona horaria correcta manualmente.'),
    sugerir('Hacer clic en ''Sincronizar ahora'' en la configuracion de hora.').

% Regla: Programas que no responden
diagnosticar(Descripcion, programas_no_responden) :-
    (   contiene_sintoma(Descripcion, 'un programa no responde');
        contiene_sintoma(Descripcion, 'aplicacion colgada');
        contiene_sintoma(Descripcion, 'programa no se cierra')
    ),
    sugerir('Presionar Ctrl + Shift + Esc para abrir el Administrador de Tareas.'),
    sugerir('Seleccionar el programa que no responde en la lista de procesos.'),
    sugerir('Hacer clic en el boton ''Finalizar tarea''.').

% Regla: Archivos adjuntos que no se abren
diagnosticar(Descripcion, adjuntos_no_abren) :-
    (   contiene_sintoma(Descripcion, 'archivos adjuntos no abren');
        contiene_sintoma(Descripcion, 'error al abrir archivo adjunto');
        contiene_sintoma(Descripcion, 'no puedo ver un archivo recibido')
    ),
    sugerir('Verificar que tiene instalado el programa correcto para abrir ese tipo de archivo (ej. Adobe Reader para PDF, Microsoft Office para .docx).'),
    sugerir('Intentar descargar el archivo adjunto en lugar de abrirlo directamente desde el correo.'),
    sugerir('Asegurarse de que su software de correo electronico o visor de archivos este actualizado.').

% Regla: Lentitud al iniciar
diagnosticar(Descripcion, lentitud_al_iniciar) :-
    (   contiene_sintoma(Descripcion, 'tarda mucho en iniciar');
        contiene_sintoma(Descripcion, 'inicio de Windows lento');
        contiene_sintoma(Descripcion, 'PC tarda en prender')
    ),
    sugerir('Abrir el Administrador de Tareas e ir a la pestaña ''Inicio''.'),
    sugerir('Deshabilitar los programas que no necesite que se inicien automaticamente con Windows.').

% Regla: Problemas con impresoras
diagnosticar(Descripcion, problemas_impresora) :-
    (   contiene_sintoma(Descripcion, 'impresora no imprime');
        contiene_sintoma(Descripcion, 'error de impresora');
        contiene_sintoma(Descripcion, 'PC no detecta impresora')
    ),
    sugerir('Verificar que la impresora este encendida y conectada a la PC (por cable o Wi-Fi).'),
    sugerir('Reiniciar la impresora y la PC.'),
    sugerir('Verificar que la impresora seleccionada sea la correcta en el cuadro de dialogo de impresion.').

% Regla: PC se congela o bloquea (frecuentemente)
diagnosticar(Descripcion, pc_se_congela) :-
    (   contiene_sintoma(Descripcion, 'PC se congela con frecuencia');
        contiene_sintoma(Descripcion, 'se bloquea constantemente');
        contiene_sintoma(Descripcion, 'se traba al usarla')
    ),
    sugerir('Actualizar los drivers de componentes clave como la tarjeta grafica, chipset y tarjeta de red desde la pagina del fabricante.'),
    sugerir('Ejecutar un diagnostico de la memoria RAM (ej. con la Herramienta de diagnostico de memoria de Windows).'),
    sugerir('Monitorear el uso de CPU y RAM (en el Administrador de Tareas) cuando ocurren los bloqueos para identificar programas problematicos.').

% Regla: Reinicios inesperados
diagnosticar(Descripcion, reinicios_inesperados) :-
    (   contiene_sintoma(Descripcion, 'PC se reinicia sola');
        contiene_sintoma(Descripcion, 'se apaga de repente');
        contiene_sintoma(Descripcion, 'reinicios inesperados')
    ),
    sugerir('Monitorear las temperaturas de la CPU y la tarjeta grafica (usando software como HWMonitor) para detectar sobrecalentamiento.'),
    sugerir('Verificar visualmente los ventiladores del PC para asegurarse de que giran correctamente.'),
    sugerir('Asegurarse de que los cables de la fuente de alimentacion esten bien conectados a la placa base y otros componentes.').

% Regla: Pantalla parpadea o muestra colores extraños
diagnosticar(Descripcion, pantalla_parpadea) :-
    (   contiene_sintoma(Descripcion, 'pantalla parpadea');
        contiene_sintoma(Descripcion, 'colores extraños en pantalla');
        contiene_sintoma(Descripcion, 'lineas en el monitor');
        contiene_sintoma(Descripcion, 'artefactos graficos')
    ),
    sugerir('Actualizar o reinstalar los controladores de la tarjeta grafica (descargarlos desde la pagina del fabricante - Nvidia, AMD, Intel).'),
    sugerir('Verificar que el cable del monitor (HDMI, DisplayPort, VGA) este bien conectado a la PC y al monitor.'),
    sugerir('Probar a ajustar la resolucion y la frecuencia de actualizacion de la pantalla.').

% Regla: No se puede acceder a internet (general)
diagnosticar(Descripcion, no_internet) :-
    (   contiene_sintoma(Descripcion, 'no tengo internet');
        contiene_sintoma(Descripcion, 'internet no funciona');
        contiene_sintoma(Descripcion, 'sin conexion a la red')
    ),
    sugerir('Verificar que el router y el modem esten encendidos y con las luces indicadoras normales.'),
    sugerir('Reiniciar el router y el modem.'),
    sugerir('Ejecutar el solucionador de problemas de red de Windows.'),
    sugerir('Verificar la configuracion de red en la PC (Adaptador de red, IP, DNS).').

% Regla: Archivos inaccesibles o corruptos
diagnosticar(Descripcion, archivos_inaccesibles) :-
    (   contiene_sintoma(Descripcion, 'archivos inaccesibles');
        contiene_sintoma(Descripcion, 'archivo corrupto');
        contiene_sintoma(Descripcion, 'no puedo abrir mis archivos');
        contiene_sintoma(Descripcion, 'error al leer disco')
    ),
    sugerir('Ejecutar un escaneo completo del sistema con un antivirus actualizado para descartar malware.'),
    sugerir('Utilizar la herramienta Comprobar Disco (chkdsk) de Windows para buscar errores en la unidad.'),
    concluir('Posible problema en el disco duro o infeccion de malware.').
                                                                        
% Regla: Ruido excesivo del PC debido a ventiladores
diagnosticar(Descripcion, ruido_excesivo_pc) :-
    (   contiene_sintoma(Descripcion, 'PC hace mucho ruido');
        contiene_sintoma(Descripcion, 'ruido fuerte en la computadora');
        contiene_sintoma(Descripcion, 'ventiladores ruidosos')
    ),
    \+ (contiene_sintoma(Descripcion, 'clic constante'); contiene_sintoma(Descripcion, 'raspado repetitivo')),
    sugerir('Limpiar el polvo acumulado en los ventiladores de la CPU, tarjeta grafica y la caja de la PC usando aire comprimido.'),
    sugerir('Verificar si el ruido proviene de un ventilador especifico que podria estar fallando.').

% Regla: Ruido excesivo del PC 
diagnosticar(Descripcion, fallo_disco_ruido) :-
    (   contiene_sintoma(Descripcion, 'ruido excesivo del PC');
        contiene_sintoma(Descripcion, 'clic constante');
        contiene_sintoma(Descripcion, 'raspado repetitivo')
    ),
    concluir('¡Atencion! Posible fallo inminente del disco duro.'),
    sugerir('Hacer una copia de seguridad de sus datos importantes inmediatamente.'),
    notificar('Es necesario acudir a un tecnico para verificar el estado del disco duro y posiblemente reemplazarlo.').


% Regla: Sobrecalentamiento 
diagnosticar(Descripcion, sobrecalentamiento) :-
    (   contiene_sintoma(Descripcion, 'PC se sobrecalienta');
        contiene_sintoma(Descripcion, 'temperatura muy alta');
        contiene_sintoma(Descripcion, 'PC quema al tocarla')
    ),
    sugerir('Limpiar el polvo de los ventiladores y disipadores (CPU, GPU).'),
    sugerir('Asegurarse de que la ventilacion de la caja de la PC no este bloqueada.'),
    concluir('El sobrecalentamiento puede dañar los componentes.').

% Regla: Puertos USB no funcionan
diagnosticar(Descripcion, puertos_usb_no_funcionan) :-
    (   contiene_sintoma(Descripcion, 'puertos USB no funcionan');
        contiene_sintoma(Descripcion, 'USB no detecta nada');
        contiene_sintoma(Descripcion, 'fallo puertos USB')
    ),
    sugerir('Reiniciar la computadora.'),
    sugerir('Desinstalar y reinstalar los controladores de los ''Controladores de Bus Serie Universal'' en el Administrador de Dispositivos.'),
    sugerir('Verificar fisicamente los puertos USB en busca de daños visibles.').

% Regla: Lector de CD/DVD no reconoce discos
diagnosticar(Descripcion, lector_cd_dvd_no_funciona) :-
    (   contiene_sintoma(Descripcion, 'lector CD no funciona');
        contiene_sintoma(Descripcion, 'DVD no reconoce disco');
        contiene_sintoma(Descripcion, 'unidad optica no lee')
    ),
    sugerir('Limpiar la lente del lector con un kit de limpieza especial para CD/DVD.'),
    sugerir('Probar con varios discos diferentes que sepa que funcionan.'),
    sugerir('Actualizar o reinstalar el controlador de la unidad de CD/DVD en el Administrador de Dispositivos.'),
    concluir('Posible fallo fisico del lector.'),
    sugerir('Considerar reemplazar el lector de CD/DVD o acudir a un tecnico.').

% Regla: Errores de actualización de Windows
diagnosticar(Descripcion, errores_windows_update) :-
    (   contiene_sintoma(Descripcion, 'errores actualizacion Windows');
        contiene_sintoma(Descripcion, 'Windows Update no instala');
        contiene_sintoma(Descripcion, 'fallo al actualizar Windows')
    ),
    sugerir('Ejecutar el Solucionador de problemas de Windows Update (buscarlo en la configuracion de Windows).'),
    sugerir('Reiniciar el servicio de Windows Update (requiere pasos especificos).'),
    sugerir('Eliminar los archivos temporales de actualizacion.').

% Regla por Defecto: No es posible dar un diagnostico
diagnostico_por_defecto :-
    notificar('Lo siento, basandome en la informacion proporcionada, mi conocimiento actual no me permite dar un diagnostico o solucion especifica para este problema.'),
    sugerir('Buscar ayuda en foros especializados en hardware/software de PC.'),
    notificar('Considera seriamente contactar a un tecnico de computadoras calificado para que revise tu equipo.').
