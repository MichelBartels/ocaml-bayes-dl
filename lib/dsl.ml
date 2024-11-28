let fn = Ir.create_func

module Var = Ir.Var

let matmul a b =
  let a_shape = Ir.shape_of_var a in
  let b_shape = Ir.shape_of_var b in
  let rec prefix = function
    | [_; _] ->
        []
    | x :: xs ->
        x :: prefix xs
    | [] ->
        failwith "matmul requires at least 2 dims"
  in
  let rec common_prefix a b =
    match (a, b) with
    | [], _ | _, [] ->
        []
    | a :: a_tl, b :: b_tl when a = b ->
        a :: common_prefix a_tl b_tl
    | _ ->
        []
  in
  let batching_dims =
    List.init
      (List.length (common_prefix (prefix a_shape) (prefix b_shape)))
      Fun.id
  in
  Ir.Var.DotProduct
    ( a
    , b
    , [List.length a_shape - 1]
    , [List.length b_shape - 2]
    , batching_dims
    , batching_dims )

let ( + ) a b = Var.Add (a, b)

let ( - ) a b = Var.Subtract (a, b)

let ( * ) a b = Var.Multiply (a, b)

let ( / ) a b = Var.Divide (a, b)

let exp a = Var.Exponential a

let abs a = Var.Abs a

let compare dir a b = Var.Compare (a, dir, b)

let ( = ) a = compare Ir.Eq a

let ( <> ) a = compare Ir.Ne a

let ( >= ) a = compare Ir.Ge a

let ( > ) a = compare Ir.Gt a

let ( <= ) a = compare Ir.Le a

let ( < ) a = compare Ir.Lt a

let full value shape = Tensor.full value shape |> Tensor.to_ir

let full_f32 value = full (F32 value)

let full_i1 value = full (I1 value)

let full_like value var =
  Ir.shape_of_var var |> Tensor.full value |> Tensor.to_ir

let ones : type a. a Ir.tensor Ir.ValueType.t -> a Ir.tensor Ir.Var.t = function
  | Tensor_type (shape, F32) ->
      full (F32 1.) shape
  | Tensor_type (shape, I1) ->
      full (I1 false) shape
  | Tensor_type (shape, I64) ->
      full (I64 1) shape

let ones_like t = ones (Ir.ValueType.of_var t)

let zeros : type a. a Ir.tensor Ir.ValueType.t -> a Ir.tensor Ir.Var.t =
  function
  | Tensor_type (shape, F32) ->
      full (F32 0.) shape
  | Tensor_type (shape, I1) ->
      full (I1 false) shape
  | Tensor_type (shape, I64) ->
      full (I64 0) shape

let zeros_like t = zeros (Ir.ValueType.of_var t)

let norm mean stddev shape =
  Var.Random
    ( Ir.ValueType.Tensor_type (shape, F32)
    , mean
    , stddev
    , Tensor.from_int_list shape |> Tensor.to_ir
    , Normal )

let uniform low high shape =
  Var.Random
    ( Ir.ValueType.Tensor_type (shape, F32)
    , low
    , high
    , Tensor.from_int_list shape |> Tensor.to_ir
    , Uniform )

let sum axis x =
  let (Tensor_type (shape, t)) = Ir.ValueType.of_var x in
  let size = List.nth shape axis in
  let ones = ones (Tensor_type ([size], t)) in
  Var.DotProduct (x, ones, [axis], [0], [], [])

let mean axis x =
  let (Tensor_type (shape, _)) = Ir.ValueType.of_var x in
  let size = List.nth shape axis in
  let fact = full_f32 (1.0 /. float_of_int size) [size] in
  Var.DotProduct (x, fact, [axis], [0], [], [])

let transpose var permutation =
  let shape = Ir.shape_of_var var in
  if Stdlib.(List.length permutation <> List.length shape) then
    failwith "Permutation length must match tensor rank" ;
  if
    not
      Stdlib.(
        List.sort compare permutation = List.init (List.length shape) Fun.id )
  then failwith "Invalid permutation" ;
  Var.Transpose (var, permutation)
