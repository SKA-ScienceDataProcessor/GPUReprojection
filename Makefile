PTXASFLAGS=-Xptxas -v,-abi=no,-dlcm=cg

ifeq ($(DEBUG),1)
	CFLAGS += -g -G -lineinfo
endif
#CFLAGS += PTXASFLAGS

reproject: reproject.cu
	nvcc $(CFLAGS) -o reproject reproject.cu
