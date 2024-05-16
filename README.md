## PyconUS talk 2024: Guía para “Fine-Tuning” local de modelos Open Source en Español [Spanish]
### Guide to Fine-tuning Open Source LLM in local set-up.

### Descripción
En el mundo actual, los modelos de lenguaje grandes (LLM, en inglés) están revolucionando cómo interactuamos con la tecnología, permitiendo tener conversaciones, organizar datos, redactar textos, y otras actividades con mínimo esfuerzo humano.

Es probable que al usar algún LLM hayas recibido respuestas incorrectas ¿a qué se debe eso? Durante el entrenamiento de estos modelos, suelen ingerir grandes cantidades de texto sin etiquetar de fuentes como libros, páginas web, foros, los cuales desarrollan un gran entendimiento de conocimiento pero carecen de conocimientos específicos. Por este motivo ajustar modelos (“Fine-Tuning”, en inglés) que han sido pre-entrenados con este gran corpus de datos es crucial para: (1) obtener mejor rendimiento en la calidad de respuestas, y (2) ajustar el modelo a un dominio específico al proporcionar textos específicos para que puedan especializarse.

Entonces, ¿Por qué es necesario entender el “Fine-Tuning” en modelos locales? Dentro de los diversos motivos, uno de los más relevantes es la privacidad de datos. Puesto que al hacer el proceso de “Fine-Tuning” localmente se puede enseñar al modelo datos que son privados, como datos personales, datos clínicos, información confidencial de empresas, etc.

En esta charla, los asistentes aprenderán paso a paso cómo modelos LLM Open Source, como Mixtral-8x22B-v0.1, Mistral-7B (multi lenguaje), bloom-7b u otros modelos, son opciones muy interesantes para aprender a realizar “Fine-Tuning” y especializar modelo para el dominio específico. Además, se compartirá el rol de Python del proceso, la aplicación de módulos externos para tener una implementación simple, para realizar “Fine-Tuning” de LLMs.
