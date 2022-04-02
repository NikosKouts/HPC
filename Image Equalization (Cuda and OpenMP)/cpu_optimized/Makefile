all: main.o contrast-enhancement.o histogram-equalization.o
	nvcc -g -O4 main.o contrast-enhancement.o histogram-equalization.o -Xcompiler -fopenmp
 
main.o: main.cu
	nvcc -c -g -O4 main.cu -o main.o

contrast-enhancement.o: contrast-enhancement.cu
	nvcc -c -g -O4 contrast-enhancement.cu -o contrast-enhancement.o

histogram-equalization.o: histogram-equalization.cu
	nvcc -c -g -O4 histogram-equalization.cu -o histogram-equalization.o

run:
	./a.out ../Images/fort.pgm fort_out.pgm
	./a.out ../Images/planet_surface.pgm planet_surface_out.pgm
	./a.out ../Images/ship.pgm ship_out.pgm
	./a.out ../Images/uth.pgm uth_out.pgm
	./a.out ../Images/x_ray.pgm x_ray_out.pgm
	./a.out ../Images/planet_surface_big.pgm planet_surface_big_out.pgm

small: 
	./a.out ../Images/uth.pgm uth_out.pgm
	diff uth_out.pgm ../golden/uth_out.pgm

medium: 
	./a.out ../Images/planet_surface_big.pgm planet_surface_big_out.pgm
	diff planet_surface_big_out.pgm ../golden/planet_surface_big_out.pgm

big:
	./a.out ../Images/planet_surface_big.pgm planet_surface_big_out.pgm
	diff planet_surface_big_out.pgm ../golden/planet_surface_big_out.pgm

diff: 
	diff fort_out.pgm ../golden/fort_out.pgm
	diff planet_surface_out.pgm ../golden/planet_surface_out.pgm
	diff uth_out.pgm ../golden/uth_out.pgm
	diff x_ray_out.pgm ../golden/x_ray_out.pgm
	diff ship_out.pgm ../golden/ship_out.pgm
	diff planet_surface_big_out.pgm ../golden/planet_surface_big_out.pgm
clean:
	rm *.o