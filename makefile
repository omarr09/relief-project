main: main.cpp
	mpic++ -o main ./src/main.cpp


run: main
	mpirun --oversubscribe -np $(NP) ./main $(ARG)