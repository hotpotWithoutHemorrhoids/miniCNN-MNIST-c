
# 当前目录makefile文件
CXX = g++
CC = gcc

# 编译选项
CXXFLAGS = -g -Wall 
CFLAGS = -g -Wall

# 目标文件
TARGET = cnn

# 源文件
SRCS = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRCS))
INCLUDE= -I./src


# 生成目标文件
$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $^ ${INCLUDE}


# 生成目标文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<
${OBJS}: ${SRCS}
	$(CC) $(CFLAGS) -c $< ${INCLUDE}
# 清理
clean:
	rm -f $(OBJS) $(TARGET)


# CC = gcc

# TARGET = cnn
# SRCS = cnn.c
# INCLUDE=src
# OBJS = $(SRCS:.c=.o)

# CFLAGS = -Wall -g -I${INCLUDE}

# %.o: %.c
# 	$(CC) $(CFLAGS) -c $< -o $@

# ${TARGET}: ${INCLUDE}/dataloader.h ${INCLUDE}/utils.h

# $(TARGET): $(OBJS)
# 	$(CC) $(OBJS) $(LDFLAGS) -o $@ 

# .PHONY: clean
# clean:
# 	rm -f $(OBJS) $(TARGET)
