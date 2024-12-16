open Iree_bindings
open Dsl

let sigmoid x = ones_like x / (ones_like x + exp (zeros_like x - x))

let tanh x = (exp x - exp (zeros_like x - x)) / (exp x + exp (zeros_like x - x))

let batch_size = 32

let reparametrize mean var shape =
  let eps =
    norm (zeros (Tensor_type ([], F32))) (ones (Tensor_type ([], F32))) shape
  in
  mean + (eps * var)

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let* w_mean = new_param (Tensor_type ([in_dims; out_dims], F32)) in
  let* w_var = new_param (Tensor_type ([in_dims; out_dims], F32)) in
  let* b_mean = new_param (Tensor_type ([1; out_dims], F32)) in
  let* b_var = new_param (Tensor_type ([1; out_dims], F32)) in
  let w_mean = Ir.Var.BroadcastInDim (w_mean, [batch_size]) in
  let w_var = Ir.Var.BroadcastInDim (w_var, [batch_size]) in
  let b_mean = Ir.Var.BroadcastInDim (b_mean, [batch_size]) in
  let b_var = Ir.Var.BroadcastInDim (b_var, [batch_size]) in
  let w = reparametrize w_mean w_var [batch_size; in_dims; out_dims] in
  let b = reparametrize b_mean b_var [batch_size; 1; out_dims] in
  return @@ activation (matmul x w + b)

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* z = dense ~activation:tanh 784 512 x in
  let* mean = dense ~activation:Fun.id 512 embedding_dim z in
  let* logvar = dense ~activation:Fun.id 512 embedding_dim z in
  return [mean; logvar]

let decoder z =
  let open Parameters in
  let* z = dense ~activation:tanh embedding_dim 512 z in
  let* z = dense 512 784 z in
  return z

let kl mean logvar =
  sum 0 @@ sum 0 @@ Dsl.mean 0
  @@ full_like (F32 ~-.0.5) mean
     * (ones_like mean + logvar - (mean * mean) - exp logvar)

let mse x x' = sum 0 @@ sum 0 @@ mean 0 ((x - x') * (x - x'))

let vae x =
  let open Parameters in
  let* [mean'; logvar] = encoder x in
  let z = reparametrize mean' (exp logvar) [batch_size; 1; embedding_dim] in
  let* x' = decoder z in
  let kl = kl mean' logvar in
  let mse = mse x x' in
  return (mse + kl)

let optim = Optim.sgd 0.000001

let train x = optim (vae x)

let train_compiled =
  let func =
    Parameters.create_func (Tensor_type ([batch_size; 1; 784], F32)) train
  in
  print_endline "Compiling train function..." ;
  Ir.compile func

let reconstruct x =
  let open Parameters in
  let* [mean'; logvar] = encoder x in
  let z = reparametrize mean' logvar [batch_size; 1; embedding_dim] in
  let* x' = decoder z in
  return x'

let reconstruct_compiled =
  let func =
    Parameters.create_func (Tensor_type ([batch_size; 1; 784], F32)) reconstruct
  in
  print_endline "Compiling reconstruct function..." ;
  Ir.compile func

let decoder_compiled =
  let func =
    Parameters.create_func
      (Tensor_type ([batch_size; 1; embedding_dim], F32))
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

(* let main = *)
(*   Backpropagate.diff [Var; Var] (fun [a; b] -> *)
(*       matmul a (Ir.Var.BroadcastInDim (b, [4])) ) *)

(* let main = *)
(*   Ir.create_func *)
(*     (List_type [Tensor_type ([4; 2; 1], F32); Tensor_type ([1; 2], F32)]) *)
(*     main *)

(* let main_compiled = Ir.compile main *)

(* let () = *)
(*   print_endline main_compiled ; *)
(*   Compile.compile main_compiled "main.vmfb" *)
