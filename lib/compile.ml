open C.Functions
open C.Type
open Ctypes

let set_hal_target session target =
  let flag = "--iree-hal-target-device=" ^ target in
  let flag_ptr = allocate string flag in
  C.Functions.session_set_flags session 1 flag_ptr

let assert_no_error = function
  | Some err ->
      let msg = error_get_message err in
      error_destroy err ; failwith msg
  | None ->
      ()

let create_out_param c_type f =
  let out_ptr = allocate_n (ptr c_type) ~count:1 in
  let err = f out_ptr in
  assert_no_error err ; !@out_ptr

let source_from_string session str =
  source_wrap_buffer session "<stdin>" str
    (String.length str |> Unsigned.Size_t.of_int)
    false
  |> create_out_param source

let create_output file = output_open_file file |> create_out_param output

let create_invocation session =
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

let ( let& ) (x, destructor) f =
  Fun.protect ~finally:(fun () -> destructor x) (fun () -> f x)

let compile str file =
  global_initialize () ;
  let& session = (session_create () |> Option.get, session_destroy) in
  set_hal_target session "cuda" ;
  let source = source_from_string session str in
  let output = create_output file in
  let& invocation = (create_invocation session, invocation_destroy) in
  parse_source invocation source ;
  invoke_pipeline invocation Std ;
  output_invocation invocation output
