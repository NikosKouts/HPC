Original Code (Path: .)
* sobel_orig.c

Optimizations that both improve the -o0 and the -fast versions (Path: .)
## How to build
1. make

## Files Included
* interchange.c
* sobel_unrolling.c
* sobel_fusion.c
* sobel_inlining.c

# Optimizations that improve the -o0 version (Path: ./comp_optim_dis_scope)
## How to build
1. cd comp_optim_dis_scope/
2. make

## Files Included
* sobel_invarian_slow.c
* sobel_elimination_slow.c
* sobel_strength_reduction_slow.c
* sobel_keywords_slow.c

# Optimizations that improve the -fast version (Path: ./comp_optim_en_scope)
## How to build
1. cd comp_optim_en_scope/
2. make

## Files Included
* sobel_invarian_fast.c
* sobel_elimination_fast.c
* sobel_strength reduction_fast.c
* sobel_keywords_fast.c

# Additional Notes
1. When bulding files on "." and "comp_optim_dis_scope/" the graph on section 4.2.1 (in report) is produced 
2. When bulding files on "." and "comp_optim_en_scope/" the graph on section 4.2.2 (in report) is produced
