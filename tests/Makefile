CC = gcc
CFlags = -lcunit -Wall 

CInclude=../include

all:
	@echo "Please specify a target: exe1 or exe2"

fc_layer_test: fc_layer_test.c
	${CC} -o fc_layer_test  fc_layer_test.c ${CFlags} -I${CInclude}

conv_test: conv_test.c
	${CC}  -I${CInclude} -o conv_test  conv_test.c ${CFlags}

clean:
	rm -rf fc_layer_test conv_test