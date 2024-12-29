open C.Functions
open C.Type
open Ctypes
open C_utils

let instance_options () =
  create_out_param (Fun.const ()) instance_options instance_options_initialize

let session_options () =
  create_out_param (Fun.const ()) session_options session_options_initialize

let assert_no_error status = assert (status_ok status)

let call session name =
  create_out_param assert_no_error call
  @@ call_initialize_by_name session
  @@ make_string_view name

let create_out_param c_type = create_out_param assert_no_error c_type

module Device = struct
  type t =
    { device: device structure Ctypes_static.ptr
    ; allocator: device_allocator structure Ctypes_static.ptr
    ; instance: instance structure Ctypes_static.ptr
    ; session: runtime_session structure Ctypes_static.ptr }

  let instance options =
    protect instance_release
    @@ create_out_param (ptr instance)
    @@ instance_create (addr options)
    @@ allocator_system ()

  let device instance device_str =
    protect device_release
    @@ create_out_param (ptr device)
    @@ instance_try_create_default_device instance (make_string_view device_str)

  let session instance options device =
    protect session_release
    @@ create_out_param (ptr runtime_session)
    @@ session_create_with_device instance (addr options) device
    @@ instance_host_allocator instance

  let make device_str =
    let options = instance_options () in
    instance_options_use_all_available_drivers (addr options) ;
    let instance = instance options in
    let device = device instance device_str in
    let session_options = session_options () in
    let session = session instance session_options device in
    let allocator = session_device_allocator session in
    {device; allocator; instance; session}
end

module Buffer = struct
  type t =
    { view: buffer_view structure Ctypes_static.ptr
    ; shape: int list
    ; device: Device.t }

  let buffer_params =
    let p = make buffer_params in
    setf p buffer_usage buffer_usage_default ;
    setf p buffer_access memory_access_all ;
    setf p buffer_memory_type memory_type_device_local ;
    p

  let make device shape view = {view; shape; device}

  let from_carray device element_type shape arr =
    let data_ptr = CArray.start arr |> to_voidp in
    let size = List.fold_left ( * ) 1 shape in
    let shape_arr =
      List.map Unsigned.Size_t.of_int shape |> CArray.of_list size_t
    in
    let rank = Unsigned.Size_t.of_int (CArray.length shape_arr) in
    let shape_ptr = CArray.start shape_arr in
    let size = size * sizeof element_type |> Unsigned.Size_t.of_int in
    { view=
        protect buffer_view_release
        @@ create_out_param (ptr buffer_view)
        @@ buffer_view_allocate_buffer_copy device.Device.device
             device.allocator rank shape_ptr element_type_float_32
             encoding_type_dense_row_major buffer_params
        @@ make_const_byte_span data_ptr size
    ; shape
    ; device }

  let from_list device shape data =
    let arr = CArray.of_list float data in
    from_carray device element_type shape arr

  let size buffer = List.fold_left ( * ) 1 buffer.shape

  let to_list buffer =
    let data = size buffer |> CArray.make float in
    let data_ptr = CArray.start data |> to_voidp in
    let size = CArray.length data * sizeof float |> Unsigned.Size_t.of_int in
    device_transfer_d2h buffer.device.device
      (buffer_view_buffer buffer.view)
      (Unsigned.Size_t.of_int 0) data_ptr size transfer_buffer_flag_default
      (infinite_timeout ())
    |> assert_no_error ;
    CArray.to_list data

  let from_tensor device tensor =
    let arr = Ir.Tensor.carray tensor in
    let element_type = Ir.Tensor.c_type @@ Ir.Tensor.kind tensor in
    let (Tensor_type (shape, _)) = Ir.Tensor.value_type tensor in
    from_carray device element_type shape arr
end

module Function = struct
  type t = {call: call structure; device: Device.t; output_shapes: int list list}

  let make device file_name function_name output_shapes =
    let session = device.Device.session in
    assert_no_error
    @@ session_append_bytecode_module_from_file session file_name ;
    let call = call session function_name in
    {call; device; output_shapes}

  let pop_output call =
    protect buffer_view_release
    @@ create_out_param (ptr buffer_view)
    @@ call_outputs_pop_front_buffer_view @@ addr call

  let call t buffers =
    List.iter
      (fun buffer ->
        assert_no_error
        @@ call_inputs_push_back_buffer_view (addr t.call) buffer.Buffer.view )
      buffers ;
    assert_no_error @@ call_invoke (addr t.call) (Unsigned.UInt32.of_int 0) ;
    let buffers =
      List.init (List.length t.output_shapes) (fun _ -> pop_output t.call)
    in
    call_reset (addr t.call) ;
    List.map2 (Buffer.make t.device) t.output_shapes buffers
end

let simple_mul () =
  let device = Device.make "local-task" in
  let function_ =
    Function.make device "simple_mul_module.vmfb" "module.simple_mul" [[2; 2]]
  in
  let x = Ir.Tensor.from_float_list [1.; 2.; 3.; 4.] in
  let y = Ir.Tensor.from_float_list [5.; 6.; 7.; 8.] in
  let z = Ir.Tensor.from_float_list [0.; 1.; 2.; 3.] in
  let buffer1 = Buffer.from_tensor device x in
  let buffer2 = Buffer.from_tensor device y in
  let results = Function.call function_ [buffer1; buffer2] in
  let result = List.hd results in
  let result = Buffer.to_list result in
  List.map string_of_float result |> String.concat ", " |> print_endline ;
  let buffer2 = Buffer.from_tensor device z in
  let results = Function.call function_ [buffer1; buffer2] in
  let result = List.hd results in
  let result = Buffer.to_list result in
  List.map string_of_float result |> String.concat ", " |> print_endline
