# Optional manual build. feedback.py builds the same library automatically
# on first import (on the machine that runs it), so this is only for dev.
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fPIC -pthread -Wall -Wextra
SRC = cpp/rm_engine.cpp cpp/analysis.cpp cpp/c_api.cpp
_build/librm.so: $(SRC) cpp/rm_engine.h cpp/util.h cpp/analysis.h
	mkdir -p _build && $(CXX) $(CXXFLAGS) -shared $(SRC) -o $@
clean:
	rm -rf _build
