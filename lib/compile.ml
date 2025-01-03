open C.Functions
open C.Type
open Ctypes
open C_utils

let set_hal_target session target =
  let flag = "--iree-hal-target-device=" ^ target in
  let flag_ptr = allocate string flag in
  session_set_flags session 1 flag_ptr

let assert_no_error = function
  | Some err ->
      let msg = error_get_message err in
      error_destroy err ; failwith msg
  | None ->
      ()

let create_out_param c_type f = create_out_param assert_no_error (ptr c_type) f

let source_from_string session str =
  source_wrap_buffer session "<stdin>" str
    (String.length str |> Unsigned.Size_t.of_int)
    false
  |> create_out_param source

let create_output file = output_open_file file |> create_out_param output

let session () = protect session_destroy @@ Option.get @@ session_create ()

let invocation session =
  protect invocation_destroy
  @@
  match invocation_create session with
  | Some invocation ->
      invocation
  | None ->
      failwith "Failed to create invocation"

let parse_source invocation source =
  let success = invocation_parse_source invocation source in
  assert success

let invoke_pipeline invocation pipeline =
  let success = invocation_pipeline invocation pipeline in
  assert success

let output_invocation invocation output =
  let err = invocation_output_vm_bytecode invocation output in
  assert_no_error err

let compile device_str str file =
  global_initialize () ;
  let session = session () in
  set_hal_target session device_str ;
  let source = source_from_string session str in
  let output = create_output file in
  let invocation = invocation session in
  parse_source invocation source ;
  invoke_pipeline invocation Std ;
  output_invocation invocation output

let cache_folder = Sys.getcwd () ^ "/.vmfb_cache"

let () =
  if not @@ Sys.file_exists cache_folder then Sys.mkdir cache_folder 0o777

let model_path str =
  let hash = Digest.string str |> Digest.to_hex in
  Filename.concat cache_folder (hash ^ ".vmfb")

let get_compiled_model device_str str =
  let path = model_path (device_str ^ str) in
  if not @@ Sys.file_exists path then compile device_str str path ;
  path
