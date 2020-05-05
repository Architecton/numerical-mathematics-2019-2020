CC=gcc 
 
CFLAGS=-c -Wall -fPIC
 
SOURCES=trapz.c monte_carlo.c
OBJECTS=$(SOURCES:.c=.o)
 
.c.o:
	$(CC) $(CFLAGS) $< -o $@ 
 
lib: $(OBJECTS)
	$(CC) -shared -fPIC -o trapz.so $(OBJECTS)
	$(CC) -shared -fPIC -o monte_carlo.so $(OBJECTS)
 
clean:
	rm *.o *.so
