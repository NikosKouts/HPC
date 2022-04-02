out: library.o main.o
	gcc -g -Wall main.o library.o -o out

main.o: main.c
	gcc -g -c -Wall main.c -o main.o

library.o: library.c
	gcc -g -c -Wall library.c -o library.o

run: out
	./out

clean:
	rm *.o out
