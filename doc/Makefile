# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

SHOW_VERSIONS = TRUE

REQUIRED="pandoc pandoc-citeproc jupyter"
RUNNOTEBOOKS=FALSE

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

check_installs:
	@for cmd in ${REQUIRED};\
	do
		if ! command -v ${cmd} 2> /dev/null;\
		then
			echo Please install ${cmd}!;\
			exit 1
		fi;\
	done;

convert-tex:
	@pandoc theory/*.tex -f latex -t rst --toc -s --bibliography=theory/bib.bib

convert-notebooks:
ifeq ($(RUNNOTEBOOKS),TRUE)
	@jupyter nbconvert user_guide/notebooks/*.ipynb --execute --to rst
else
	@jupyter nbconvert user_guide/notebooks/*.ipynb --to rst
endif

clean:
	rm -rf build/ generated/ reference/generated/
	rm -f user_guide/notebooks/*.rst
	rm -f theory/*.rst
	rm -rf user_guide/notebooks/*/

all:
	@make clean
	@make convert-tex
	@make convert-notebooks

	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)