
CYTHON ?= cython

cython_sources := $(wildcard *.pyx)
c_sources = $(cython_sources:.pyx=.c)

all : $(c_sources)

%.c : %.pyx
	$(CYTHON) $<
