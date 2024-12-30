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

module rec Device : sig
  type local_task

  type _ kind = Local_task : local_task kind

  type 'a t =
    { device: device structure Ctypes_static.ptr
    ; allocator: device_allocator structure Ctypes_static.ptr
    ; instance: instance structure Ctypes_static.ptr
    ; session: runtime_session structure Ctypes_static.ptr
    ; kind: 'a kind }

  val make : 'a kind -> 'a t

  (* val compile : 'a Ir.ValueType.t -> ('a Ir.Var.t -> 'b Ir.Var.t) ->  *)
end = struct
  type local_task

  type _ kind = Local_task : local_task kind

  type 'a t =
    { device: device structure Ctypes_static.ptr
    ; allocator: device_allocator structure Ctypes_static.ptr
    ; instance: instance structure Ctypes_static.ptr
    ; session: runtime_session structure Ctypes_static.ptr
    ; kind: 'a kind }

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

  let device_str : type a. a kind -> string = function
    | Device.Local_task ->
        "local-task"

  let make kind =
    let options = instance_options () in
    instance_options_use_all_available_drivers (addr options) ;
    let instance = instance options in
    let device = device instance @@ device_str kind in
    let session_options = session_options () in
    let session = session instance session_options device in
    let allocator = session_device_allocator session in
    {device; allocator; instance; session; kind}
end

and Buffer : sig
  type ('a, 'b, 'c) t =
    { view: buffer_view structure Ctypes_static.ptr
    ; shape: int list
    ; device: 'c Device.t
    ; kind: ('a, 'b) Ir.tensor }

  val make :
       'c Device.t
    -> int list
    -> ('a, 'b) Ir.tensor
    -> buffer_view structure Ctypes_static.ptr
    -> ('a, 'b, 'c) t

  val of_float_list :
    'a Device.t -> int list -> float list -> (Ir.f32, float, 'a) t

  val to_float_list : (Ir.f32, float, 'a) t -> float list

  val of_tensor : 'c Device.t -> ('a, 'b) Ir.Tensor.t -> ('a, 'b, 'c) t

  val to_tensor : ('a, 'b, 'c) t -> ('a, 'b) Ir.Tensor.t
end = struct
  type ('a, 'b, 'c) t =
    { view: buffer_view structure Ctypes_static.ptr
    ; shape: int list
    ; device: 'c Device.t
    ; kind: ('a, 'b) Ir.tensor }

  let buffer_params =
    let p = make buffer_params in
    setf p buffer_usage buffer_usage_default ;
    setf p buffer_access memory_access_all ;
    setf p buffer_memory_type memory_type_device_local ;
    p

  let make device shape kind view = {view; shape; device; kind}

  let of_carray device element_type shape kind arr =
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
    ; device
    ; kind }

  let of_float_list device shape data =
    let arr = CArray.of_list float data in
    of_carray device element_type shape F32 arr

  let size buffer = List.fold_left ( * ) 1 buffer.shape

  let to_carray buffer =
    let c_type = Ir.Tensor.c_type buffer.kind in
    let data = size buffer |> CArray.make c_type in
    let data_ptr = CArray.start data |> to_voidp in
    let size = CArray.length data * sizeof c_type |> Unsigned.Size_t.of_int in
    device_transfer_d2h buffer.device.device
      (buffer_view_buffer buffer.view)
      (Unsigned.Size_t.of_int 0) data_ptr size transfer_buffer_flag_default
      (infinite_timeout ())
    |> assert_no_error ;
    data

  let to_float_list buffer = to_carray buffer |> CArray.to_list

  let of_tensor device tensor =
    let arr = Ir.Tensor.carray tensor in
    let element_type = Ir.Tensor.c_type @@ Ir.Tensor.kind tensor in
    let (Tensor_type (shape, _)) = Ir.Tensor.value_type tensor in
    of_carray device element_type shape (Ir.Tensor.kind tensor) arr

  let to_tensor buffer =
    let arr = to_carray buffer in
    let shape = buffer.shape in
    let kind = buffer.kind in
    Ir.Tensor.from_carray kind shape arr
end

and Function : sig
  type ('a, 'b, 'c) t

  val make :
       'c Device.t
    -> string
    -> string
    -> 'a Ir.ValueType.t
    -> 'b Ir.ValueType.t
    -> ('a, 'b, 'c) t

  val call :
       ('a, 'b, 'c) t
    -> ('a, 'c Value.device) Value.t
    -> ('b, 'c Value.device) Value.t
end = struct
  type ('a, 'b, 'c) t =
    {call: call structure; device: 'c Device.t; output_type: 'b Ir.ValueType.t}

  let make device file_name function_name _ output_type =
    let session = device.Device.session in
    assert_no_error
    @@ session_append_bytecode_module_from_file session file_name ;
    let call = call session function_name in
    {call; device; output_type}

  let pop_output call =
    protect buffer_view_release
    @@ create_out_param (ptr buffer_view)
    @@ call_outputs_pop_front_buffer_view @@ addr call

  let rec push_inputs :
      type a b c d. (a, b, c) t -> (d, c Value.device) Value.t -> unit =
   fun t -> function
    | Value.Device buffer ->
        assert_no_error
        @@ call_inputs_push_back_buffer_view (addr t.call) buffer.view
    | Value.[] ->
        ()
    | Value.(hd :: tl) ->
        push_inputs t hd ; push_inputs t tl

  let rec pop_outputs :
      type a b c d.
      (a, b, c) t -> d Ir.ValueType.t -> (d, c Value.device) Value.t =
   fun t -> function
    | Tensor_type (shape, kind) ->
        let buffer = pop_output t.call in
        let buffer = Buffer.make t.device shape kind buffer in
        Value.Device buffer
    | List_type (hd :: tl) ->
        Value.(pop_outputs t hd :: pop_outputs t (List_type tl))
    | List_type [] ->
        Value.[]

  let call t buffers =
    push_inputs t buffers ;
    assert_no_error @@ call_invoke (addr t.call) (Unsigned.UInt32.of_int 0) ;
    let outputs = pop_outputs t t.output_type in
    call_reset (addr t.call) ;
    outputs
end

and Value : sig
  type host = |

  type 'a device = |

  type (_, _) t =
    | Host : ('a, 'b) Ir.Tensor.t -> (('a, 'b) Ir.tensor, host) t
    | Device : ('a, 'b, 'c) Buffer.t -> (('a, 'b) Ir.tensor, 'c device) t
    | [] : (unit Ir.VarList.t, 'a) t
    | ( :: ) :
        ('a, 'b) t * ('c Ir.VarList.t, 'b) t
        -> (('a Ir.Var.t -> 'c) Ir.VarList.t, 'b) t

  val move_to_device : 'a Device.t -> ('b, 'c) t -> ('b, 'a device) t

  val move_to_host : ('a, 'b) t -> ('a, host) t
end = struct
  type host = |

  type 'a device = |

  type (_, _) t =
    | Host : ('a, 'b) Ir.Tensor.t -> (('a, 'b) Ir.tensor, host) t
    | Device : ('a, 'b, 'c) Buffer.t -> (('a, 'b) Ir.tensor, 'c device) t
    | [] : (unit Ir.VarList.t, 'a) t
    | ( :: ) :
        ('a, 'b) t * ('c Ir.VarList.t, 'b) t
        -> (('a Ir.Var.t -> 'c) Ir.VarList.t, 'b) t

  let rec move_to_device : type a b c. a Device.t -> (b, c) t -> (b, a device) t
      =
   fun device -> function
    | Host tensor ->
        Device (Buffer.of_tensor device tensor)
    | Device buffer ->
        Device (Buffer.of_tensor device @@ Buffer.to_tensor buffer)
    | [] ->
        []
    | hd :: tl ->
        move_to_device device hd :: move_to_device device tl

  let rec move_to_host : type a b. (a, b) t -> (a, host) t = function
    | Host tensor ->
        Host tensor
    | Device buffer ->
        Host (Buffer.to_tensor buffer)
    | [] ->
        []
    | hd :: tl ->
        move_to_host hd :: move_to_host tl
end

let simple_mul () =
  let device = Device.make Local_task in
  let function_ =
    Function.make device "simple_mul_module.vmfb" "module.simple_mul"
      (List_type [Tensor_type ([2; 2], F32); Tensor_type ([2; 2], F32)])
      (Tensor_type ([2; 2], F32))
  in
  let x = Ir.Tensor.from_float_list [1.; 2.; 3.; 4.] in
  let y = Ir.Tensor.from_float_list [5.; 6.; 7.; 8.] in
  let inputs = Value.[Host x; Host y] in
  let inputs = Value.move_to_device device inputs in
  let result = Function.call function_ inputs in
  let (Host result) = Value.move_to_host result in
  print_endline @@ Ir.Tensor.to_string result
