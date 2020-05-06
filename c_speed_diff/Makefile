CC = g++
CFLAGS = -Wall -Wextra -Winline -Wno-deprecated

main: funcs.c
	$(CC) $(CFLAGS) -fPIC -shared -o funcs.so funcs.c

clean:
	rm -f *.so
