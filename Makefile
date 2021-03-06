PTXASFLAGS=-Xptxas -v,-abi=no,-dlcm=cg

ifeq ($(DEBUG),1)
	CFLAGS += -g -G -lineinfo
else
	CFLAGS += $(PTXASFLAGS)
endif
CFLAGS += -arch=sm_35

ifdef DATATYPE
	CFLAGS += -DDATATYPE=$(DATATYPE) -DDATATYPE2=$(DATATYPE)2
else
	CFLAGS += -DDATATYPE=double -DDATATYPE2=double2
endif
ifdef DATATYPE_INTERP
	CFLAGS += -DDATATYPE_INTERP=$(DATATYPE_INTERP) -DDATATYPE_INTERP2=$(DATATYPE_INTERP)2
else
	DATATYPE_INTERP=DATATYPE
endif

ifeq ($(FASTMATH),0)
else
	CFLAGS += -use_fast_math
endif
ifeq ($(INTERP),LINEAR)
	CFLAGS += -D__INTERP_LINEAR
endif
ifeq ($(TEXTURE),1)
	CFLAGS += -D__USE_TEX
ifeq ($(DATATYPE_INTERP),double)
	#Double precision cannot use texture interpolation
	CFLAGS += -D__SLOW_TEX
endif
ifeq ($(SLOW_TEX),1)
	CFLAGS += -D__SLOW_TEX
endif
endif

reproject: reproject.cu
ifdef ERR
	$(error $(ERR))
endif
	nvcc $(CFLAGS) -o reproject reproject.cu
