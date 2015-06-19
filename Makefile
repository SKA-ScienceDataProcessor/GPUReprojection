ifeq ($(DEBUG),1)
	CFLAGS += -g -G -lineinfo
endif

reproject:
	nvcc $(CFLAGS) -o reproject reproject.cu
