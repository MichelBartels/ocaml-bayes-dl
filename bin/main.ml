open Iree_bindings
open Dsl

let sigmoid x = ones_like x / (ones_like x + exp (zeros_like x - x))

let tanh x = (exp x - exp (zeros_like x - x)) / (exp x + exp (zeros_like x - x))

let batch_size = 512

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

let kl mean logvar =
  Dsl.sum 0 @@ Dsl.mean 0
  @@ full_like (F32 ~-.0.5) mean
     * (ones_like mean + logvar - (mean * mean) - exp logvar)

let mse x x' = sum 0 (mean 0 ((x - x') * (x - x')))

let vae x =
  let open Parameters in
  let* [mean'; logvar] = encoder x in
  let z = reparametrize mean' logvar in
  let* x' = decoder z in
  let kl = kl mean' logvar in
  let mse = mse x x' in
  return (mse + kl)

let optim f =
  let open Parameters in
  let* [params; grad; loss] = grad_and_value f in
  let scale_grad : type a. a Ir.tensor Ir.Var.t -> a Ir.tensor Ir.Var.t =
   fun grad ->
    match Ir.ValueType.of_var grad with
    | Tensor_type (_, F32) ->
        grad * full_like (F32 0.02) grad
    | Tensor_type (_, I1) ->
        grad
    | Tensor_type (_, I64) ->
        grad
  in
  return [loss; Ir.Var.map2 {fn= (fun p g -> p - scale_grad g)} params grad]

let train x = optim (vae x)

let train_compiled =
  let func =
    Parameters.create_func (Tensor_type ([batch_size; 784], F32)) train
  in
  print_endline "Compiling train function..." ;
  Ir.compile func

let reconstruct x =
  let open Parameters in
  let* [mean'; logvar] = encoder x in
  let z = reparametrize mean' logvar in
  let* x' = decoder z in
  return x'

let reconstruct_compiled =
  let func =
    Parameters.create_func (Tensor_type ([batch_size; 784], F32)) reconstruct
  in
  print_endline "Compiling reconstruct function..." ;
  Ir.compile func

let decoder_compiled =
  let func =
    Parameters.create_func
      (Tensor_type ([batch_size; embedding_dim], F32))
      decoder
  in
  print_endline "Compiling decoder function..." ;
  Ir.compile func

let () =
  print_endline train_compiled ;
  Compile.compile train_compiled "train.vmfb" ;
  print_endline reconstruct_compiled ;
  Compile.compile reconstruct_compiled "reconstruct.vmfb" ;
  print_endline decoder_compiled ;
  Compile.compile decoder_compiled "decoder.vmfb"
