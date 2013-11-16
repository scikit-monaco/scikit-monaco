
PYTHON := python

all : rings.png rings.pdf scatter.png scatter.pdf import_plot.png import_plot.pdf import_integrand.pdf import_integrand.png

rings.png rings.pdf : draw_circles.py
	$(PYTHON) draw_circles.py

scatter.png scatter.pdf : draw_scatter.py
	$(PYTHON) draw_scatter.py

import_plot.png import_plot.pdf : draw_import.py
	$(PYTHON) draw_import.py

import_integrand.png import_integrand.pdf : import_surf.py
	$(PYTHON) import_surf.py
