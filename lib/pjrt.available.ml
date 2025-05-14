(* This file will be selected and copied to pjrt.ml if PJRT bindings are installed *)

let ( let* ) = Option.bind

let try_load () =
  let* path = Sys.getenv_opt "PJRT_PATH" in
  let metal = Option.is_some @@ Sys.getenv_opt "METAL" in
  if metal then Metal.enable () ;
  Some (Pjrt_bindings.make ~caching:(not metal) path)

let try_load_with_prompt () =
  print_endline
    "No default backend specified via environment variable. Do you want to use \
     PJRT? (y/n)" ;
  let answer = read_line () in
  if String.lowercase_ascii answer <> "y" then None
  else (
    print_endline
      "Please specify the path to the PJRT plugin (e.g. \
       /path/to/pjrt_plugin.so):" ;
    let path = read_line () in
    print_endline "Is this a Metal backend? (y/n)" ;
    let answer = read_line () in
    let metal = String.lowercase_ascii answer = "y" in
    if metal then Metal.enable () ;
    Some (Pjrt_bindings.make ~caching:(not metal) path) )
