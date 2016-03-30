#
# run like so:
#   env LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH ./bin/cvg -h
#
CUDA_HOME = /usr/local/cuda-7.5
NVCC = $(CUDA_HOME)/bin/nvcc
DEFINES = -DNO_CPU_BLAS -DNO_WINDOWS

bin/cvg: cvg/main.o cvg/util.o cvg/gpu_blas_test.o
	@mkdir -p bin
	$(NVCC) $^ -o $@ -lcudart -lcublas

# NVCC seems to need C++11, but host compiler needs C++0x
cvg/gpu_blas_test.o: %.o : %.cu %.h
	$(NVCC) $(DEFINES) -std=c++11 -c $< -o $@

# FIXME find Intel MKL libraries for linux
#cpu_blas_test.o: %.o : %.cpp %.h
cvg/util.o: %.o : %.cpp %.h
cvg/main.o: %.o : %.cpp
cvg/%.o:
	$(NVCC) $(DEFINES) -Xcompiler "-std=c++0x" -c $< -o $@

clean:
	rm -f bin/cvg cvg/*.o
