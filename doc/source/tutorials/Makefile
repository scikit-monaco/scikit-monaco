
PYTHON := python
MV := mv

images = rings.png rings.pdf scatter.png scatter.pdf import_plot.png import_plot.pdf import_integrand.pdf import_integrand.png

all : $(addprefix img/,$(images))

img/rings.png img/rings.pdf : draw_circles.py
	$(PYTHON) draw_circles.py
	@if [ ! -d img/ ] ; then mkdir img/ ; fi
	$(MV) rings.png rings.pdf img/

img/scatter.png img/scatter.pdf : draw_scatter.py
	$(PYTHON) draw_scatter.py
	@if [ ! -d img/ ] ; then mkdir img/ ; fi
	$(MV)  scatter.png scatter.pdf img/

img/import_plot.png img/import_plot.pdf : draw_import.py
	$(PYTHON) draw_import.py
	@if [ ! -d img/ ] ; then mkdir img/ ; fi
	$(MV) import_plot.png import_plot.pdf img/

img/import_integrand.png img/import_integrand.pdf : import_surf.py
	$(PYTHON) import_surf.py
	@if [ ! -d img/ ] ; then mkdir img/ ; fi
	$(MV) import_integrand.png import_integrand.pdf img/

clean:
	rm -rf img/*
