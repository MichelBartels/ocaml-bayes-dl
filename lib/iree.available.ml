(* This file will be selected and copied to iree.ml if IREE bindings are installed *)

let ( let* ) = Option.bind

let backend_of_string backend =
  match String.lowercase_ascii backend with
  | "cpu" ->
      Some (Iree_bindings.make Cpu)
  | "vulkan" ->
      Some (Iree_bindings.make Vulkan)
  | "cuda" ->
      Some (Iree_bindings.make Cuda)
  | backend ->
      Printf.eprintf "Unsupported backend: %s" backend ;
      None

let try_load () =
  let* backend = Sys.getenv_opt "IREE_BACKEND" in
  backend_of_string backend

let try_load_with_prompt () =
  print_endline
    "No default backend specified via environment variable. Do you want to use \
     IREE? (y/n)" ;
  let answer = read_line () in
  if String.lowercase_ascii answer <> "y" then None
  else (
    print_endline "Please specify the backend to use (cpu, vulkan, cuda):" ;
    let backend = read_line () in
    backend_of_string backend )
