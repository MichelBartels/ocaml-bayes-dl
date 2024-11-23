open Iree_bindings
open Dsl

let sigmoid x = ones_like x / (ones_like x + exp (zeros_like x - x))

let tanh x = (exp x - exp (zeros_like x - x)) / (exp x + exp (zeros_like x - x))

let batch_size = 128

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let* w = new_param (Tensor_type ([in_dims; out_dims], F32)) in
  let* b = new_param (Tensor_type ([out_dims], F32)) in
  let b = Ir.Var.BroadcastInDim (b, [batch_size]) in
  return @@ activation (matmul x w + b)

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* z = dense ~activation:tanh 784 512 x in
  let* mean = dense ~activation:Fun.id 512 embedding_dim z in
  let* logvar = dense ~activation:Fun.id 512 embedding_dim z in
  return [mean; logvar]

let reparametrize mean logvar =
  let eps =
    norm
      (zeros (Tensor_type ([], F32)))
      (ones (Tensor_type ([], F32)))
      [batch_size; embedding_dim]
  in
  mean + (eps * exp logvar)

let decoder z =
  let open Parameters in
  let* z = dense ~activation:tanh embedding_dim 512 z in
  let* z = dense 512 784 z in
  return z

let mse x y =
  let diff = x - y in
  mean 0 (mean 0 (diff * diff))

let kl mean logvar =
  full_like (F32 0.5) mean
  * (ones_like mean + logvar - (mean * mean) - exp logvar)

let vae x =
  let open Parameters in
  let* [mean'; logvar] = encoder x in
  let z = reparametrize mean' logvar in
  let* x' = decoder z in
  let kl = kl mean' logvar in
  let kl = mean 0 (mean 0 kl) in
  let mse = mse x x' in
  return (kl + mse)

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

let main x = optim (vae x)

let main_compiled =
  let func =
    Parameters.create_func (Tensor_type ([batch_size; 784], F32)) main
  in
  print_endline "Compiling..." ;
  Ir.compile func

let () =
  print_endline main_compiled ;
  Compile.compile main_compiled "out.vmfb"
