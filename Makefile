
all: two

xor: xor.cpp
	g++ xor.cpp -o xor -O3 -DNHID=32
	
two: 2layer.cpp
	gcc -O3 2layer.cpp -o two -lm -lstdc++ -std=c++11 -Wall

deps:
	sudo apt-get update
	sudo apt-get install g++ make
