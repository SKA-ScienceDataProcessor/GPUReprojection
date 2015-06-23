ifeq ($(DEBUG),1)
	CFLAGS += -g -G -lineinfo
endif

reproject: reproject.cu
	nvcc $(CFLAGS) -o reproject reproject.cu
