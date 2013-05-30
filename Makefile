PROGRAMS=\
  time-math-x86-64 \
  time-math-x86-64-DAZ \
  time-math-x86-64-FTZ \
  time-math-x86-64-DAZ-FTZ \
  time-math-x86-32-x87 \
  time-math-x86-32-sse2 \
  time-math-x86-32-sse2-DAZ \
  time-math-x86-32-sse2-FTZ \
  time-math-x86-32-sse2-DAZ-FTZ \
  time-math-arm-soft \
  time-math-arm-vfp \
  time-math-arm-vfp-FZ \
  time-math-arm-vfpv3 \
  time-math-arm-vfpv3-FZ

all: $(PROGRAMS) runner

time-math-x86-64:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -o time-math-x86-64

time-math-x86-64-DAZ:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -DX86_ENABLE_DAZ -o time-math-x86-64-DAZ

time-math-x86-64-FTZ:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -DX86_ENABLE_FTZ -o time-math-x86-64-FTZ

time-math-x86-64-DAZ-FTZ:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -DX86_ENABLE_DAZ -DX86_ENABLE_FTZ -o time-math-x86-64-DAZ-FTZ

time-math-x86-32-x87:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -m32 -o time-math-x86-32-x87

time-math-x86-32-sse2:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -m32 -msse2 -mfpmath=sse -o time-math-x86-32-sse2

time-math-x86-32-sse2-DAZ:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -m32 -msse2 -mfpmath=sse -DX86_ENABLE_DAZ -o time-math-x86-32-sse2-DAZ

time-math-x86-32-sse2-FTZ:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -m32 -msse2 -mfpmath=sse -DX86_ENABLE_FTZ -o time-math-x86-32-sse2-FTZ

time-math-x86-32-sse2-DAZ-FTZ:
	g++ -O2 time-math.cpp --std=c++0x -lrt -static -m32 -msse2 -mfpmath=sse -DX86_ENABLE_DAZ -DX86_ENABLE_FTZ -o time-math-x86-32-sse2-DAZ-FTZ

time-math-arm-soft:
	/usr/bin/arm-linux-gnueabi-g++-4.6 time-math.cpp --std=c++0x -lrt -static -mfloat-abi=soft -o time-math-arm-soft

time-math-arm-vfp:
	/usr/bin/arm-linux-gnueabi-g++-4.6 time-math.cpp --std=c++0x -lrt -static -mfloat-abi=softfp -mfpu=vfp -o time-math-arm-vfp

time-math-arm-vfp-FZ:
	/usr/bin/arm-linux-gnueabi-g++-4.6 time-math.cpp --std=c++0x -lrt -static -mfloat-abi=softfp -mfpu=vfp -DARM_ENABLE_FZ -o time-math-arm-vfp-FZ

time-math-arm-vfpv3:
	/usr/bin/arm-linux-gnueabi-g++-4.6 time-math.cpp --std=c++0x -lrt -static -mfloat-abi=softfp -mfpu=vfpv3 -o time-math-arm-vfpv3

time-math-arm-vfpv3-FZ:
	/usr/bin/arm-linux-gnueabi-g++-4.6 time-math.cpp --std=c++0x -lrt -static -mfloat-abi=softfp -mfpu=vfpv3 -DARM_ENABLE_FZ -o time-math-arm-vfpv3-FZ

runner:
	echo 'PROGRAMS="$(PROGRAMS)"'                            > run.sh
	echo 'for p in $$PROGRAMS; do echo $$p; ./$$p > $$p.txt; done' >> run.sh
	chmod a+x run.sh

clean:
	rm $(PROGRAMS) run.sh

push:
	for p in $(PROGRAMS) run.sh; do adb push $$p /data/local/tmp; done
