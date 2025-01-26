
# 当前目录makefile文件
CC = gcc

# 编译选项
CXXFLAGS = -g -o0-Wall -lm 
CFLAGS = -g -Wall -O3 -lm -ffast-math

CInclude=-I./include

# 目标文件
TARGET_CNN = cnn
TARGET_NN = nn

# nn 的目标文件 和源文件
NN_SRCS = nn.c
NN_OBJS = ${NN_SRCS:.c=.o}

# CNN 的目标文件和源文件
CNN_SRCS= cnn.c
CNN_OBJS = ${CNN_SRCS:.c=.o}

# 默认目标
all: $(TARGET_NN) $(TARGET_NN)

# 生成 nn 可执行文件
${TARGET_NN}: ${NN_OBJS}
	${CC} -o $@ $^ ${CInclude} ${CFLAGS}

# 生成 cnn 可执行文件
$(TARGET_CNN): $(CNN_OBJS)
	${CC}  -o $@ $^ ${CInclude} ${CFLAGS}

# 生成 .o 文件
%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@ ${CInclude}

# 清理生成的文件
clean:
	rm -f ${NN_OBJS} ${CNN_OBJS} ${TARGET_CNN} ${TARGET_NN}

.PHONY: all clean