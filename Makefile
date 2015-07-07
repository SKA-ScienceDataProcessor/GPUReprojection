PTXASFLAGS=-Xptxas -v,-abi=no,-dlcm=cg

ifeq ($(DEBUG),1)
	CFLAGS += -g -G -lineinfo
endif
CFLAGS += $(PTXASFLAGS)
CFLAGS += -arch=sm_35

ifdef DATATYPE
	CFLAGS += -DDATATYPE=$(DATATYPE) -DDATATYPE2=$(DATATYPE)2
else
	CFLAGS += -DDATATYPE=double -DDATATYPE2=double2
endif
ifdef DATATYPE_INTERP
	CFLAGS += -DDATATYPE_INTERP=$(DATATYPE_INTERP) -DDATATYPE_INTERP2=$(DATATYPE_INTERP)2
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
endif

reproject: reproject.cu
ifeq ($(TEXTURE),1)
	ifeq ($(DATATYPE),double)
		$(error DATATYPE double is incompatible with TEXTURE=1)
	endif
endif
	nvcc $(CFLAGS) -o reproject reproject.cu
