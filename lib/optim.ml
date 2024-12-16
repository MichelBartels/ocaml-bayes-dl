open Dsl

let sgd lr f =
  let open Parameters in
  let* [params; grad; loss] = grad_and_value f in
  let scale_grad : type a. a Ir.tensor Ir.Var.t -> a Ir.tensor Ir.Var.t =
   fun grad ->
    match Ir.ValueType.of_var grad with
    | Tensor_type (_, F32) ->
        grad * full_like (F32 lr) grad
    | Tensor_type (_, I1) | Tensor_type (_, I64) ->
        failwith "can only optimize floating point grads"
  in
  return [loss; Ir.Var.map2 {fn= (fun p g -> p - scale_grad g)} params grad]
