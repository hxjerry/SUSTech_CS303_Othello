XSIMD_DIR = /usr/include/xsimd
CXXFLAGS += -I$(XSIMD_DIR) -march=native -O2 -std=c++17 -Wall
CXX = g++

main: main.o ../bitboard.o
	$(CXX) $(CXXFLAGS) -o main main.o ../bitboard.o