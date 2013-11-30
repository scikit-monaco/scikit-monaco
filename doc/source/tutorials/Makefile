
PYTHON := python
MV := mv

images = rings.png rings.pdf scatter.png scatter.pdf import_plot.png import_plot.pdf import_integrand.pdf import_integrand.png

all : $(addprefix build/,$(images))

build/rings.png build/rings.pdf : draw_circles.py
	$(PYTHON) draw_circles.py
	@if [ ! -d build/ ] ; then mkdir build/ ; fi
	$(MV) rings.png rings.pdf build/

build/scatter.png build/scatter.pdf : draw_scatter.py
	$(PYTHON) draw_scatter.py
	@if [ ! -d build/ ] ; then mkdir build/ ; fi
	$(MV)  scatter.png scatter.pdf build/

build/import_plot.png build/import_plot.pdf : draw_import.py
	$(PYTHON) draw_import.py
	@if [ ! -d build/ ] ; then mkdir build/ ; fi
	$(MV) import_plot.png import_plot.pdf build/

build/import_integrand.png build/import_integrand.pdf : import_surf.py
	$(PYTHON) import_surf.py
	@if [ ! -d build/ ] ; then mkdir build/ ; fi
	$(MV) import_integrand.png import_integrand.pdf build/

clean:
	rm -rf build/*
