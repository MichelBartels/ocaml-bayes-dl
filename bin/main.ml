open Iree_bindings
open Dsl

(* let sigmoid x = ones_like x / (ones_like x + exp (zeros_like x - x)) *)

let dense x =
  let open Parameters in
  let* w = new_param (Tensor_type ([1; 1], F32)) in
  let* b = new_param (Tensor_type ([1], F32)) in
  return (matmul x w + b)

let mse x y =
  let diff = x - y in
  diff * diff

let optim f =
  let open Parameters in
  let* [params; grad; loss] = grad_and_value f in
  let scale_grad : type a. a Ir.tensor Ir.Var.t -> a Ir.tensor Ir.Var.t =
   fun grad ->
    match Ir.ValueType.of_var grad with
    | Tensor_type (_, F32) ->
        grad * full_like (F32 0.1) grad
    | Tensor_type (_, I1) ->
        grad
    | Tensor_type (_, I64) ->
        grad
  in
  return [loss; Ir.Var.map2 {fn= (fun p g -> p - scale_grad g)} params grad]

let f x y =
  let open Parameters in
  let* z = dense x in
  return (mse z y)

let f x y = optim (f x y)

let f =
  Parameters.create_func
    (List_type [Tensor_type ([2; 1], F32); Tensor_type ([2; 1], F32)])
    (fun [x; y] -> f x y)

let f_compiled = Ir.compile f

let () =
  print_endline f_compiled ;
  Compile.compile f_compiled "out.vmfb"
