let fn = Ir.create_func

module Var = Ir.Var

let ( + ) a b = Var.Add (a, b)

let ( - ) a b = Var.Subtract (a, b)

let ( * ) a b = Var.Multiply (a, b)

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

let ones_like : type a. a Ir.tensor Ir.Var.t -> a Ir.tensor Ir.Var.t =
 fun var ->
  match Ir.ValueType.of_var var with
  | Tensor_type (shape, F32) ->
      full (F32 1.) shape
  | Tensor_type (shape, I1) ->
      full (I1 true) shape
  | Tensor_type (shape, I64) ->
      full (I64 1) shape

let zeros_like : type a. a Ir.tensor Ir.Var.t -> a Ir.tensor Ir.Var.t =
 fun var ->
  match Ir.ValueType.of_var var with
  | Tensor_type (shape, F32) ->
      full (F32 0.) shape
  | Tensor_type (shape, I1) ->
      full (I1 false) shape
  | Tensor_type (shape, I64) ->
      full (I64 0) shape

let norm mean stddev shape =
  Ir.tag
    (Var.Random
       ( Ir.ValueType.Tensor_type (shape, F32)
       , mean
       , stddev
       , Tensor.from_int_list shape |> Tensor.to_ir
       , Normal ) )

let uniform low high shape =
  Ir.tag
    (Var.Random
       ( Ir.ValueType.Tensor_type (shape, F32)
       , low
       , high
       , Tensor.from_int_list shape |> Tensor.to_ir
       , Uniform ) )
