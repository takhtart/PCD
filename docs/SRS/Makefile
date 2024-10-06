# Makefile to build requirements standard plan
# author: Sébastien Mosser
# /!\ Run make check_env to check that your setup is correct

.DEFAULT_GOAL := pdf
INDEX   = index.adoc          # Root file for the requirement plan
THEME   = _style/mcmaster.yml # Location of the PDF stylesheet

####
# Building PlantUML models
####

MODELS := $(shell find * -type f -name "*.puml")
GRAPHS := $(shell find * -type f -name "*.dot")

SVGs = $(MODELS:.puml=.svg) $(GRAPHS:.dot=.svg)  

# transform a puml file into it's svg representation
%.svg: %.puml
	plantuml -tsvg ./$<

%.svg: %.dot
	dot -Tsvg ./$< > $@

####
# Building the HTML version
####

html: $(INDEX) $(SVGs)
	asciidoctor $(INDEX)

####
# Building the PDF version
####

pdf: $(INDEX) $(SVGs)
	asciidoctor-pdf --theme $(THEME) $(INDEX)

####
# Cleaning (and clobbering) 
####

clean:
	rm -rf *.html *.pdf 

clobber: clean
	rm -rf $(SVGs)

####
# Checking environment
####

check_env:
	@which asciidoctor
	@which asciidoctor-pdf
	@which plantuml
	@which dot
	@plantuml -testdot



