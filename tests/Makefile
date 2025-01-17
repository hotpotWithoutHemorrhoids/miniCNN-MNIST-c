CC = gcc
CFlags = -lcunit -Wall 

CInclude=../include

all:
	@echo "Please specify a target: exe1 or exe2"

fc_layer_test: fc_layer_test.c
	${CC} ${CFlags} -I${CInclude} -o fc_layer_test  fc_layer_test.c

conv_test: conv_test.c
	${CC} ${CFlags} -I${CInclude} -o conv_test  conv_test.c

clean:
	rm -rf fc_layer_test conv_test