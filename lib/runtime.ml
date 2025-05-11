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

    let make buffer shape kind =
      let buffer = {buffer; shape; kind} in
      buffer

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
          let v = Buffer.to_tensor buffer in
          Device.collect_buffer buffer.buffer ;
          E v
      | [] ->
          []
      | hd :: tl ->
          to_host_value hd :: to_host_value tl

    let collect : type a. a t -> unit =
     fun t ->
      let list = List.to_any_list t in
      Stdlib.List.iter
        (fun (Any buffer) -> Device.collect_buffer buffer.buffer)
        list
  end

  module Function : sig
    type ('a, 'b) t

    val make :
      Device.program -> 'a Ir.ValueType.t -> 'b Ir.ValueType.t -> ('a, 'b) t

    val call :
      ('a, 'b) t -> ?collect:bool -> 'a DeviceValue.t -> 'b DeviceValue.t
  end = struct
    type ('a, 'b) t =
      {program: Device.program; output_type: 'b Ir.ValueType.t; num_outputs: int}

    let make program _ output_type =
      let num_outputs = Ir.ValueType.List.num_elements output_type in
      {program; output_type; num_outputs}

    let rec flatten_inputs : type a.
        Device.buffer list -> a DeviceValue.t -> Device.buffer list =
     fun acc -> function
      | E buffer ->
          buffer.buffer :: acc
      | [] ->
          acc
      | hd :: tl ->
          flatten_inputs (flatten_inputs acc hd) tl

    let rec nest_outputs : type a.
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

    let call t ?(collect = true) buffers =
      let buffers = flatten_inputs [] buffers in
      let buffers = List.rev buffers in
      let outputs =
        Device.execute ~num_outputs:t.num_outputs t.program buffers
      in
      if collect then List.iter Device.collect_buffer buffers ;
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
    let input_type = Ir.ValueType.List.[input_type; E ([], F32)] in
    let func =
      Ir.create_func input_type (fun [x; E seed] ->
              let seed = Dsl.convert U64 seed in
          Random.handler
            (fun () ->
              let y = f x in
              let seed = Dsl.convert F32 (Random.current_seed ()) in
              Ir.Var.List.[y; E seed] )
            seed )
    in
    let output_type = Ir.ValueType.of_vars func.outputs in
    let func_str = Ir.compile func in
    (*let func_str = {|
    func.func @main(%1 : tensor<f32>) -> (tensor<f32>) {
%2 = "stablehlo.convert"(%1) : (tensor<f32>) -> (tensor<ui64>)
%3 = "stablehlo.optimization_barrier"(%2) : (tensor<ui64>) -> (tensor<ui64>)
%4 = "stablehlo.multiply"(%2, %3) : (tensor<ui64>, tensor<ui64>) -> (tensor<ui64>)
%5 = "stablehlo.convert"(%4) : (tensor<ui64>) -> (tensor<f32>)
"func.return"(%5) : (tensor<f32>) -> ()
}
    |} in*)
    let model_path = model_path func_str in
    let program = Device.compile ~path:model_path func_str in
    let func = Function.make program input_type output_type in
    let seed =
      ref @@ DeviceValue.of_host_value
      @@ HostValue.E (Ir.Tensor.scalar F32 0.0)
    in
    fun ?collect inputs ->
      let [y; seed'] = Function.call func ?collect [inputs; !seed] in
      seed := seed' ;
      y
end
