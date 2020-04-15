CC=gcc 
 
CFLAGS=-c -Wall -fPIC
 
SOURCES=trapz.c 
OBJECTS=$(SOURCES:.c=.o)
 
.c.o:
	$(CC) $(CFLAGS) $< -o $@ 
 
lib: $(OBJECTS)
	$(CC) -shared -fPIC -o trapz.so $(OBJECTS)
 
clean:
	rm *.o *.so
