(lang dune 3.16)

(using ctypes 0.3)

(name ocaml_bayes_dl)

(generate_opam_files true)

(source
 (github MichelBartels/ocaml-bayes-dl))

(authors "Michel Bartels")

(maintainers "Michel Bartels")

(license LICENSE)

(documentation https://url/to/documentation)

(package
 (name ocaml_bayes_dl)
 (depends
  ocaml
  dune
  ctypes
  integers
  progress
  device_api
  graphics
  domainslib
  stb_image_write
  ppx_inline_test
  ppx_expect)
 (tags
  (topics "to describe" your project)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/reference/dune-project/index.html
