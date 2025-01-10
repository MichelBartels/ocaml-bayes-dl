let cache_folder = Sys.getcwd () ^ "/.vmfb_cache"

let () =
  if not @@ Sys.file_exists cache_folder then Sys.mkdir cache_folder 0o777

module HostValue = struct
  module List = Hlist.Make (struct
    type ('a, 'b) t = ('a, 'b) Ir.Tensor.t
  end)

  include List

  let rec zeros : type a. a Ir.ValueType.t -> a t = function
    | E (shape, kind) ->
        E (Ir.Tensor.zeros kind shape)
    | hd :: tl ->
        zeros hd :: zeros tl
    | [] ->
        []

  let value_type t =
    let open Hlist.Map (List) (Ir.ValueType.List) in
    map {f= Ir.ValueType.of_tensor} t
end

module Make (Device : Device_api.S) = struct
  module Buffer = struct
    type ('a, 'b) t =
      {buffer: Device.buffer; shape: int list; kind: ('a, 'b) Ir.Tensor.kind}

    let make buffer shape kind = {buffer; shape; kind}

    let of_tensor tensor =
      let shape = Ir.Tensor.shape tensor in
      let kind = Ir.Tensor.kind tensor in
      let buffer = Device.tensor_to_buffer tensor in
      {buffer; shape; kind}

    let to_tensor {buffer; shape; kind} =
      Device.buffer_to_tensor ~shape kind buffer
  end

  module DeviceValue = struct
    module List = Hlist.Make (struct
      type ('a, 'b) t = ('a, 'b) Buffer.t
    end)

    include List

    let value_type t =
      let open Hlist.Map (List) (Ir.ValueType.List) in
      map {f= (fun buffer -> (buffer.shape, buffer.kind))} t

    let rec of_host_value : type a. a HostValue.t -> a t = function
      | E tensor ->
          E (Buffer.of_tensor tensor)
      | [] ->
          []
      | hd :: tl ->
          let hd = of_host_value hd in
          let tl = of_host_value tl in
          hd :: tl

    let rec to_host_value : type a. a t -> a HostValue.t = function
      | E buffer ->
          E (Buffer.to_tensor buffer)
      | [] ->
          []
      | hd :: tl ->
          to_host_value hd :: to_host_value tl
  end

  module Function = struct
    type ('a, 'b) t =
      {program: Device.program; output_type: 'b Ir.ValueType.t; num_outputs: int}

    let make program output_type =
      let num_outputs = Ir.ValueType.List.num_elements output_type in
      {program; output_type; num_outputs}

    let rec flatten_inputs :
        type a. Device.buffer list -> a DeviceValue.t -> Device.buffer list =
     fun acc -> function
      | E buffer ->
          buffer.buffer :: acc
      | [] ->
          acc
      | hd :: tl ->
          flatten_inputs (flatten_inputs acc hd) tl

    (* let rec collect_inputs : type a b. (a, b device) Value.t -> unit = function *)
    (*   | Value.Device buffer -> *)
    (*       Buffer.collect buffer *)
    (*   | Value.[] -> *)
    (*       () *)
    (*   | Value.(hd :: tl) -> *)
    (*       collect_inputs hd ; collect_inputs tl *)

    let rec nest_outputs :
        type a.
           Device.buffer list
        -> a Ir.ValueType.t
        -> a DeviceValue.t * Device.buffer list =
     fun buffers -> function
      | E (shape, kind) ->
          let buffer, buffers =
            match buffers with
            | hd :: tl ->
                (hd, tl)
            | [] ->
                failwith "nest_outputs: not enough buffers"
          in
          let buffer = Buffer.make buffer shape kind in
          (DeviceValue.E buffer, buffers)
      | hd :: tl ->
          let hd, buffers = nest_outputs buffers hd in
          let tl, buffers = nest_outputs buffers tl in
          (hd :: tl, buffers)
      | [] ->
          ([], buffers)

    let call t buffers =
      let buffers = flatten_inputs [] buffers in
      let buffers = List.rev buffers in
      let outputs =
        Device.execute ~num_outputs:t.num_outputs t.program buffers
      in
      let outputs, _ = nest_outputs outputs t.output_type in
      outputs
  end

  let hash fun_str =
    let str = Device.identifier ^ fun_str in
    Digest.string str |> Digest.to_hex

  let model_path str =
    let hash = hash str in
    cache_folder ^ "/" ^ hash

  let compile input_type f =
    let input_type = Ir.ValueType.List.[input_type; E ([], Ir.Tensor.U64)] in
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
    print_endline func_str ;
    let model_path = model_path func_str in
    let program =
      if Sys.file_exists model_path then Device.load ~path:model_path
      else Device.compile_and_store ~program:func_str ~path:model_path
    in
    let func = Function.make program output_type in
    let seed =
      ref @@ DeviceValue.of_host_value
      @@ HostValue.E (Ir.Tensor.scalar U64 @@ Unsigned.UInt64.of_int 0)
    in
    fun inputs ->
      let [y; seed'] = Function.call func [inputs; !seed] in
      seed := seed' ;
      y
end
