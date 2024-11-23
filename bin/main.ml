open Iree_bindings
open Dsl

let sigmoid x = ones_like x / (ones_like x + exp (zeros_like x - x))

let batch_size = 128

let dense in_dims out_dims x =
  let open Parameters in
  let* w = new_param (Tensor_type ([in_dims; out_dims], F32)) in
  let* b = new_param (Tensor_type ([out_dims], F32)) in
  let b = Ir.Var.BroadcastInDim (b, [batch_size]) in
  return @@ sigmoid (matmul x w + b)

let mse x y =
  let diff = x - y in
  mean 0 (mean 0 (diff * diff))

let optim f =
  let open Parameters in
  let* [params; grad; loss] = grad_and_value f in
  let scale_grad : type a. a Ir.tensor Ir.Var.t -> a Ir.tensor Ir.Var.t =
   fun grad ->
    match Ir.ValueType.of_var grad with
    | Tensor_type (_, F32) ->
        grad * full_like (F32 0.0001) grad
    | Tensor_type (_, I1) ->
        grad
    | Tensor_type (_, I64) ->
        grad
  in
  return [loss; Ir.Var.map2 {fn= (fun p g -> p - scale_grad g)} params grad]

let f x y =
  let open Parameters in
  let* z = dense 784 128 x in
  let* z = dense 128 10 z in
  return (mse z y)

let f x y = optim (f x y)

let f =
  Parameters.create_func
    (List_type
       [ Tensor_type ([batch_size; 784], F32)
       ; Tensor_type ([batch_size; 10], F32) ] )
    (fun [x; y] -> f x y)

let f_compiled = Ir.compile f

let () =
  print_endline f_compiled ;
  Compile.compile f_compiled "out.vmfb"
