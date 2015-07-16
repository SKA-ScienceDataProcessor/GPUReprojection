# GPUReprojection
Reprojection on the GPU

#Build options

FASTMATH=1,0            Enables fast math processing for trigonometric functions. 
                        ON by default, FASTMATH=0 to disable.
DEBUG=0,1               Compiles with debug flags
DATATYPE=double,float   Specifies precision. Default is double
DATATYPE_INTERP=        Specifies precision for interpolation. Default is DATATYPE.
         double,float
INTERP=LINEAR           linear interpolation. Default is bicubic spline
TEXTURE=0,1             Use texture to read the image. By default, this also enables
                            a special use of hardware accelerated texture interpolation.
                            See below.
SLOW_TEX=0,1            If texture read is enabled, this flag can be used to disable
                            the use of texture interpolation. Instead, texture will 
                            read on-grid values and interpolate in the kernel itself.

