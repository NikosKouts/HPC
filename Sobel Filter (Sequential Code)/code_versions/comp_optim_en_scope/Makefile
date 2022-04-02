all: invariant_no_comp_optim.out invariant_optim.out elimination_no_comp_optim.out elimination_optim.out strength_reduction_no_comp_optim.out strength_reduction_optim.out keywords_no_comp_optim.out keywords_optim.out

#Compile Invariant
invariant_no_comp_optim.out: sobel_invariant_fast.c
	icc -Wall -O0 sobel_invariant_fast.c -o invariant_no_comp_optim.out

invariant_optim.out: sobel_invariant_fast.c
	icc -Wall -fast sobel_invariant_fast.c -o invariant_optim.out

#Compile Elimination
elimination_no_comp_optim.out: sobel_elimination_fast.c
	icc -Wall -O0 sobel_elimination_fast.c -o elimination_no_comp_optim.out

elimination_optim.out: sobel_elimination_fast.c
	icc -Wall -fast sobel_elimination_fast.c -o elimination_optim.out

#Compile Strength Reduction
strength_reduction_no_comp_optim.out: sobel_strength_reduction_fast.c
	icc -Wall -O0 sobel_strength_reduction_fast.c -o strength_reduction_no_comp_optim.out

strength_reduction_optim.out: sobel_strength_reduction_fast.c
	icc -Wall -fast sobel_strength_reduction_fast.c -o strength_reduction_optim.out

#Compile Invariant
keywords_no_comp_optim.out: sobel_keywords_fast.c
	icc -Wall -O0 sobel_keywords_fast.c -o keywords_no_comp_optim.out

keywords_optim.out: sobel_keywords_fast.c
	icc -Wall -fast sobel_keywords_fast.c -o keywords_optim.out

clean: 
	rm *.out