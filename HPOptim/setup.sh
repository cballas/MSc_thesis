# Author: Julien Hoachuck
# Copyright 2015, Julien Hoachuck, All rights reserved.

luajit -b ../model.lua model.o
gcc -w -c -Wall -Wl,-E -fpic cluaf.c -lluajit -lluaT -lTH -lm -ldl -I./compiled -I/usr/include/lua5.1 -L./compiled
#gcc -shared cluaf.o model.o -L$TorchPath/install/lib -lluajit -lluaT -lTH -lm -ldl -Wl,-E -o libcluaf.so
gcc -shared cluaf.o model.o -L../../../torch/install/lib -lluajit -lluaT -lTH -L$TorchPath/install/lib -lm -ldl -Wl,-E -o libcluaf.so
