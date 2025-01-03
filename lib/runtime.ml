open C.Functions
open C.Type
open Ctypes
open C_utils

let instance_options () =
  create_out_param (Fun.const ()) instance_options instance_options_initialize

let session_options () =
  create_out_param (Fun.const ()) session_options session_options_initialize

let assert_no_error status =
  if not @@ status_ok status then (
    let msg_ptr = allocate_n (ptr char) ~count:1 in
    let len_ptr = allocate_n size_t ~count:1 in
    let allocator = allocator_system () in
    status_to_string status (addr allocator) msg_ptr len_ptr ;
    failwith @@ Ctypes_std_views.string_of_char_ptr !@msg_ptr )

let call session name =
  create_out_param assert_no_error call
  @@ call_initialize_by_name session
  @@ make_string_view name

let create_out_param c_type = create_out_param assert_no_error c_type

module rec Device : sig
  type local_task

  type cuda

  type _ kind = Local_task : local_task kind | Cuda : cuda kind

  type 'a t =
    { device: device structure Ctypes_static.ptr
    ; allocator: device_allocator structure Ctypes_static.ptr
    ; instance: instance structure Ctypes_static.ptr
    ; session: runtime_session structure Ctypes_static.ptr
    ; kind: 'a kind }

  val make : 'a kind -> 'a t

  val compile :
       'a Device.t
    -> 'b Ir.ValueType.t
    -> ('b Ir.Var.t -> 'c Ir.Var.t)
    -> ('b, 'a Value.device) Value.t
    -> ('c, 'a Value.device) Value.t
end = struct
  type local_task

  type cuda

  type _ kind = Local_task : local_task kind | Cuda : cuda kind

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
    | Device.Cuda ->
        "cuda"

  let compile_device_str : type a. a kind -> string = function
    | Device.Local_task ->
        "vmvx"
    | Device.Cuda ->
        "cuda"

  let make kind =
    let options = instance_options () in
    instance_options_use_all_available_drivers (addr options) ;
    let instance = instance options in
    let device = device instance @@ device_str kind in
    let session_options = session_options () in
    let session = session instance session_options device in
    let allocator = session_device_allocator session in
    {device; allocator; instance; session; kind}

  let compile device input_type f =
    let input_type = Ir.ValueType.List.[input_type; E ([], Ir.U64)] in
    let func =
      Ir.create_func input_type (fun [x; E seed] ->
          Random.handler
            (fun () ->
              let y = f x in
              Ir.Var.List.[y; E (Random.current_seed ())] )
            seed )
    in
    let output_type = Ir.ValueType.of_vars func.outputs in
    let func_str = Ir.compile func in
    let compiled =
      Compile.get_compiled_model (compile_device_str device.kind) func_str
    in
    let func = Function.make device compiled func.name input_type output_type in
    let seed =
      Value.Host (Ir.Tensor.scalar_u64 "0")
      |> Value.move_to_device device
      |> ref
    in
    fun inputs ->
      let [y; seed'] = Function.call func [inputs; !seed] in
      seed := seed' ;
      y
end

and Buffer : sig
  type ('a, 'b) t =
    { view: buffer_view structure Ctypes_static.ptr
    ; shape: int list
    ; device: 'b Device.t
    ; kind: 'a Ir.tensor
    ; mutable collected: bool }

  val make :
       'b Device.t
    -> int list
    -> 'a Ir.tensor
    -> buffer_view structure Ctypes_static.ptr
    -> ('a, 'b) t

  val of_float_list :
    'a Device.t -> int list -> float list -> (Ir.f32 * float, 'a) t

  val to_float_list : (Ir.f32 * float, 'a) t -> float list

  val of_tensor : 'b Device.t -> 'a Ir.Tensor.t -> ('a, 'b) t

  val to_tensor : ('a * 'b, 'c) t -> ('a * 'b) Ir.Tensor.t

  val move : 'a Device.t -> ('b, 'c) t -> ('b, 'a) t

  val collect : ('a, 'b) t -> unit
end = struct
  type ('a, 'b) t =
    { view: buffer_view structure Ctypes_static.ptr
    ; shape: int list
    ; device: 'b Device.t
    ; kind: 'a Ir.tensor
    ; mutable collected: bool }

  let buffer_params =
    let p = make buffer_params in
    setf p buffer_usage buffer_usage_default ;
    setf p buffer_access memory_access_all ;
    setf p buffer_memory_type memory_type_device_local ;
    p

  let iree_element_type : type a. a Ir.tensor -> Unsigned.uint32 = function
    | F32 ->
        element_type_float_32
    | F64 ->
        element_type_float_64
    | I1 ->
        element_type_bool_8 (* TODO: need to check if this is correct *)
    | I64 ->
        element_type_int_64
    | U32 ->
        element_type_uint_32
    | U64 ->
        element_type_uint_64

  let collect buffer =
    if not buffer.collected then (
      (* if false then ( *)
      buffer.collected <- true ;
      buffer_view_release buffer.view )

  let make device shape kind view =
    protect collect {view; shape; device; kind; collected= false}

  let of_carray device element_type shape kind arr =
    let data_ptr = CArray.start arr |> to_voidp in
    let size = List.fold_left ( * ) 1 shape in
    let shape_arr =
      List.map Unsigned.Size_t.of_int shape |> CArray.of_list size_t
    in
    let rank = Unsigned.Size_t.of_int (CArray.length shape_arr) in
    let shape_ptr = CArray.start shape_arr in
    let size = size * sizeof element_type |> Unsigned.Size_t.of_int in
    protect collect
      { view=
          create_out_param (ptr buffer_view)
          @@ buffer_view_allocate_buffer_copy device.Device.device
               device.allocator rank shape_ptr (iree_element_type kind)
               encoding_type_dense_row_major buffer_params
          @@ make_const_byte_span data_ptr size
      ; shape
      ; device
      ; kind
      ; collected= false }

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

  let to_any_carray : type a b. (a, b) t -> Ir.Tensor.any_carray =
   fun buffer ->
    let (Any c_type) = Ir.Tensor.any_c_type buffer.kind in
    let data = size buffer |> CArray.make c_type in
    let data_ptr = CArray.start data |> to_voidp in
    let size = CArray.length data * sizeof c_type |> Unsigned.Size_t.of_int in
    device_transfer_d2h buffer.device.device
      (buffer_view_buffer buffer.view)
      (Unsigned.Size_t.of_int 0) data_ptr size transfer_buffer_flag_default
      (infinite_timeout ())
    |> assert_no_error ;
    Any data

  let to_float_list buffer = to_carray buffer |> CArray.to_list

  let of_tensor : type a b. a Device.t -> b Ir.Tensor.t -> (b, a) t =
   fun device tensor ->
    let (Any arr) = Ir.Tensor.any_carray tensor in
    let (Any element_type) = Ir.Tensor.any_c_type @@ Ir.Tensor.kind tensor in
    let shape, _ = Ir.Tensor.value_type tensor in
    of_carray device element_type shape (Ir.Tensor.kind tensor) arr

  let move : type a b c. a Device.t -> (b, c) t -> (b, a) t =
   fun device buffer ->
    let (Any arr) = to_any_carray buffer in
    let (Any element_type) = Ir.Tensor.any_c_type buffer.kind in
    of_carray device element_type buffer.shape buffer.kind arr

  let to_tensor buffer =
    let arr = to_carray buffer in
    let shape = buffer.shape in
    let kind = buffer.kind in
    collect buffer ;
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
    let call = call session @@ "module." ^ function_name in
    {call; device; output_type}

  let pop_output call =
    create_out_param (ptr buffer_view)
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

  let rec collect_inputs : type a b. (a, b Value.device) Value.t -> unit =
    function
    | Value.Device buffer ->
        Buffer.collect buffer
    | Value.[] ->
        ()
    | Value.(hd :: tl) ->
        collect_inputs hd ; collect_inputs tl

  let rec pop_outputs :
      type a b c d.
      (a, b, c) t -> d Ir.ValueType.t -> (d, c Value.device) Value.t =
   fun t -> function
    | E (shape, kind) ->
        let buffer = pop_output t.call in
        let buffer = Buffer.make t.device shape kind buffer in
        Value.Device buffer
    | hd :: tl ->
        let hd = pop_outputs t hd in
        let tl = pop_outputs t tl in
        Value.(hd :: tl)
    | [] ->
        Value.[]

  let call t buffers =
    push_inputs t buffers ;
    assert_no_error @@ call_invoke (addr t.call) (Unsigned.UInt32.of_int 0) ;
    collect_inputs buffers ;
    let outputs = pop_outputs t t.output_type in
    call_reset (addr t.call) ;
    outputs
end

and Value : sig
  type host = |

  type 'a device = |

  type (_, _) t =
    | Host : 'a Ir.Tensor.t -> ('a Ir.Var.u Hlist.element, host) t
    | Device : ('a, 'b) Buffer.t -> ('a Ir.Var.u Hlist.element, 'b device) t
    | [] : (unit Hlist.hlist, 'a) t
    | ( :: ) :
        ('a, 'b) t * ('c Hlist.hlist, 'b) t
        -> (('a -> 'c) Hlist.hlist, 'b) t

  val move_to_device : 'a Device.t -> ('b, 'c) t -> ('b, 'a device) t

  val move_to_host :
       (('a * 'b) Ir.Var.u Hlist.element, 'c) t
    -> (('a * 'b) Ir.Var.u Hlist.element, host) t

  val value_type : ('a, 'b) t -> 'a Ir.ValueType.t

  val zeros : 'a Ir.ValueType.t -> ('a, host) t
end = struct
  type host = |

  type 'a device = |

  type (_, _) t =
    | Host : 'a Ir.Tensor.t -> ('a Ir.Var.u Hlist.element, host) t
    | Device : ('a, 'b) Buffer.t -> ('a Ir.Var.u Hlist.element, 'b device) t
    | [] : (unit Hlist.hlist, 'a) t
    | ( :: ) :
        ('a, 'b) t * ('c Hlist.hlist, 'b) t
        -> (('a -> 'c) Hlist.hlist, 'b) t

  let rec move_to_device : type a b c. a Device.t -> (b, c) t -> (b, a device) t
      =
   fun device -> function
    | Host tensor ->
        Device (Buffer.of_tensor device tensor)
    | Device buffer ->
        Device (Buffer.move device buffer)
    | [] ->
        []
    | hd :: tl ->
        let hd = move_to_device device hd in
        let tl = move_to_device device tl in
        hd :: tl

  let move_to_host :
      type a b c.
         ((a * b) Ir.Var.u Hlist.element, c) t
      -> ((a * b) Ir.Var.u Hlist.element, host) t = function
    | Host tensor ->
        Host tensor
    | Device buffer ->
        Host (Buffer.to_tensor buffer)
  (* | [] -> *)
  (*     [] *)
  (* | hd :: tl -> *)
  (*     move_to_host hd :: move_to_host tl *)

  let rec value_type : type a b. (a, b) t -> a Ir.ValueType.t = function
    | Host tensor ->
        E (Ir.Tensor.value_type tensor)
    | Device buffer ->
        E (buffer.shape, buffer.kind)
    | [] ->
        []
    | hd :: tl ->
        value_type hd :: value_type tl

  let rec zeros : type a. a Ir.ValueType.t -> (a, host) t = function
    | E (shape, kind) ->
        Host (Ir.Tensor.zeros kind shape)
    | hd :: tl ->
        zeros hd :: zeros tl
    | [] ->
        []
end
