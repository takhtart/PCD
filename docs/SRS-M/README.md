# Handbook Template for Requirements Document

- Author: Sébastien Mosser
- Version: 23.08

## Disclaimer

This template structure is an AsciiDoctor implementation of the four "books" described by [Bertrand Meyer](https://bertrandmeyer.com/bio/) in his book [Handbook of Requirements and Business Analysis](https://se.inf.ethz.ch/requirements/).

A copy of the chapter describing how the structure should be filled from a bird-eye point of view is available in [Chapter 3](./plan.pdf). for more information, please refer to (1) the book or (2) your lectures on requirements engineering & project management.

The template is provided in a way that fits McMaster University course portfolio (Faculty of Engineering, CAS department). It is explicitely designed as a support for CS/SE 3RA3.

## How to use it

### Required software

  - Asciidoctor (https://docs.asciidoctor.org/asciidoctor/latest/install/)
  - If using UML diagrams: PlantUML (https://plantuml.com/starting)

A makefile is provided to automate the compilation of the requirements document. To check if your installation is correct, use the `make check_env` command:

```
mosser@azrael tpl % make check_env
/opt/homebrew/bin/asciidoctor
/opt/homebrew/bin/asciidoctor-pdf
/opt/homebrew/bin/plantuml
/opt/homebrew/bin/dot
Dot version: dot - graphviz version 8.1.0 (20230707.0739)
Installation seems OK. File generation OK
```

### Configure your document (`metadata.adoc`)

The `metadata.adoc` file contains all the metadata for your document:

  - The first line defines the authors (`Name <email>`), separated by semicolons. The template supports up to six authors.
  - `:project-title:` is the name of your project. You can refer to it in your text by using `{project-title}`.
  - `:course-number:` refers to the course code associated to this delivery (e.g., `3RA3`, `4G06`)
  - `:course-term:` refers to the term associated to this course (e.g., `Fall 2023`)
  - `:rev-number:` is the _REV_ ision _NUMBER_ of the document (should start at 1)
  - `:env-draft:` is a boolean variable to include the _hints_ about the content of each section in the section. Comment this line (`// :env-draft:`) to remove the hints in your final delivery

### Contribute to your document

1. The core contents of the document is located in the `parts` directory. Each _book_ is in its own subdirectory. 
    - For example, if you want to edit section _G.3 Expected Benefits_, you have to edit the file named `parts/goals/G3.adoc`
2. If you create a UML model in a file with extension `.puml` in the `models` directory, the makefile will compile it into a vectorial format (SVG) so that you can include it in your document.
    - Look for example at `models/domain_model.puml` and `models/use_cases.puml`, included in sections XX and YY (respectively)
    - You have to include the stylesheet (`!include ../_style/puml.style`) to use McMaster colour palette. If not, you'll use the default PlantUML theme.
3. If you want to add appendices to your document, we recommend you adding the related `adoc` files in the `appendix` directory, and to include them in your `index.adoc` file
    - You have to use the `[appendix]` annotation so that Asciidoctor knows you're declaring an appendix section. Look how _Appendix A_ is defined in `appendix/A_example.adoc` and included in `index.adoc`.



### Build the requirement documents

To create the PDF document, simply run `make pdf` (this is the default target). It'll create a file named `index.pdf` with the compiled version of your document.

```
mosser@azrael tpl % make pdf
asciidoctor-pdf --theme _style/mcmaster.yml index.adoc        
```


If you want to create an HTML version of your document, run `make html`. It'll create an `index.html` file containing your document

```
mosser@azrael tpl % make html
asciidoctor index.adoc            
```

## Iterative Development


Writing requirements is an incremental and iterative process. As such, you should not write your document from A to Z in one shot, but instead think of multiple intermediate milestones.

Based on the experience accumulated by Dr. Paige & Dr. Mosser on teaching CS/SE 3RA3, we recomend the following milestones:

- **Milestone #1**: Focus on Goals & Stakeholders
  + Goals: G.1, G.2, G.3, G.4, G.7
  + Environment: E.1, E.5, E.6
  + Project: P.6, P.7
  + System: ø
- **Milestone #2**: Focus on Functionalities & Scenario 
  + Goals: G.5, G.6
  + Environment: E.2, E.3, E.4, and update E.1 with a domain model
  + Project: ø
  + System: S.1, S.2, S.3, S.4, S.6
- **Milestone #3**: Focus on finalizing the document
  + Goals: ø
  + Environment: ø
  + Project: S.5, update S.6 with acceptance tests
  + System: P.1, P.2, P.3, P.4, P.5, update P.6 with threats.


## Modifying the stylesheets

### Images

- The `_style/logo.png` file is McMaster CAS department official logo
- The `_style/license.png` file is automatically included at the bottom of each page

### McMaster AsciiDoctor PDF style

The file `_style/mcmaster.yml` extends the default PDF exportation style to match McMaster color palette and logos.

### PlantUML stylesheet

The stylesheet is defined in the file named `_style/puml.style`. It refers to classical UML elements used in requirements document, using McMaster _marroon_ and _gold_ color palette.

PlantUML documentation is now really helpful to define theme, so it was defined in a _"monkey see, monkey do"_ way based on examples available in PlantMUL source code repository: https://github.com/plantuml/plantuml/blob/master/themes/ 