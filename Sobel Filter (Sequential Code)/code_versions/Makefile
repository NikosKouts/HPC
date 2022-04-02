all: orig_no_comp_optim.out orig_optim.out interchange_no_comp_optim.out interchange_optim.out unrolling_no_comp_optim.out unrolling_optim.out fusion_no_comp_optim.out fusion_optim.out inlining_no_comp_optim.out inlining_optim.out

#Compile Original Code
orig_no_comp_optim.out: sobel_orig.c
	icc -Wall -O0 sobel_orig.c -o orig_no_comp_optim.out

orig_optim.out: sobel_orig.c
	icc -Wall -fast sobel_orig.c -o orig_optim.out

#Compile Interchange
interchange_no_comp_optim.out: sobel_interchange.c
	icc -Wall -O0 sobel_interchange.c -o interchange_no_comp_optim.out

interchange_optim.out: sobel_interchange.c
	icc -Wall -fast sobel_interchange.c -o interchange_optim.out

#Compile Unrolling
unrolling_no_comp_optim.out: sobel_unrolling.c
	icc -Wall -O0 sobel_unrolling.c -o unrolling_no_comp_optim.out

unrolling_optim.out: sobel_unrolling.c
	icc -Wall -fast sobel_unrolling.c -o unrolling_optim.out

#Compile Fusion
fusion_no_comp_optim.out: sobel_fusion.c
	icc -Wall -O0 sobel_fusion.c -o fusion_no_comp_optim.out

fusion_optim.out: sobel_fusion.c
	icc -Wall -fast sobel_fusion.c -o fusion_optim.out

#Compile Fusion
inlining_no_comp_optim.out: sobel_inlining.c
	icc -Wall -O0 sobel_inlining.c -o inlining_no_comp_optim.out

inlining_optim.out: sobel_inlining.c
	icc -Wall -fast sobel_inlining.c -o inlining_optim.out

clean:
	rm *.out