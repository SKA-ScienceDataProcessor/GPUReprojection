# GPUReprojection
Reprojection on the GPU

#Build options

FASTMATH=1,0            Enables fast math processing for trigonometric functions. 
                        ON by default, FASTMATH=0 to disable.
DEBUG=0,1               Compiles with debug flags
DATATYPE=double,float   Specifies precision. Default is double
DATATYPE_INTERP=        Specifies precision for interpolation. Default is DATATYPE.
         double,float
