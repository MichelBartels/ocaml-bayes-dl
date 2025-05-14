# OCaml-Bayes-DL
== Build instructions
To use this package, you need to install [Device-API](https://github.com/MichelBartels/device-api) and a backend. For the backend you can choose between [PJRT](https://github.com/MichelBartels/ocaml-pjrt) (recommended) and [IREE](https://github.com/MichelBartels/ocaml-iree).

To install the dependencies, please follow the instructions in the repository linked.

Afterwards, you can install this package (assuming that OCaml opam are installed) with the following command:

``` sh
opam pin add ocaml_bayes_dl https://github.com/MichelBartels/ocaml-bayes-dl.git
```

== Specifying a backend
If you are including this library in your project, you need to construct the device using a backend of your choice which you then need to pass to the runtime functor. For example, a PJRT backend can be configured like this:

``` ocaml
module Device = ( val Pjrt_bindings.make "path/to/pjrt/plugin.so" )
module Runtime = Runtime.Make (Device)
```

If you are running the tests, a backend is determined at runtime. To specify the backend either pass the `PJRT_PATH` enviroment variable or the `IREE_BACKEND` environment variable. If you are using PJRT and metal, you also need to set the `METAL` enviroment variable to enable workarounds due to Apple's incomplete PJRT metal plugin.

## Running the VAE example
After installing the package, you can run the command `ocaml_bayes_dl` which will automatically download the MNIST dataset and store samples throughout the training process in a `samples` folder. The backend can either be specified using environment variables as described or interactively when running the program.
