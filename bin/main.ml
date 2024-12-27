open Iree_bindings
open Dsl

(* let tanh x = (exp x -@ exp (0.0 -.< x)) /@ (exp x +@ exp (0.0 -.< x)) *)

let () = Printexc.record_backtrace true

(* let func = *)
(*   Parameters.create_func (List_type []) (fun [] -> *)
(*       Parameters.return @@ [Random.normal_f32 [4; 4]; Random.uniform_f32 [4; 4]] ) *)

(* let ir = Ir.compile func *)

(* let () = print_endline ir *)

(* let () = Compile.compile ir "sample.vmfb" *)

let sigmoid x = (tanh (x /.> 2.0) +.> 1.0) /.> 2.0

let batch_size = 32

let reparametrize mean var shape =
  (* let eps = Random.normal_f32 shape in *)
  let eps =
    norm (zeros (Tensor_type ([], F32))) (ones (Tensor_type ([], F32))) shape
  in
  mean +@ (eps *@ var)

let kl mean logvar var =
  sum [0]
  @@ Dsl.mean [0; 1]
  @@ (-0.5 *.< (1.0 +.< logvar -@ (mean *@ mean) -@ var))

let bayesian_parameter shape =
  let open Parameters in
  let* mean = new_param (Tensor_type (shape, F32)) in
  let* var = new_param (Tensor_type (shape, F32)) in
  let mean = Ir.Var.BroadcastInDim (mean, [batch_size]) in
  let var = Ir.Var.BroadcastInDim (var, [batch_size]) in
  let loss = kl mean (ln (var *.> 1000.)) (var *.> 1000.) in
  return @@ [reparametrize mean var (batch_size :: shape); loss]

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let* [w; w_loss] = bayesian_parameter [in_dims; out_dims] in
  let* [b; b_loss] = bayesian_parameter [1; out_dims] in
  return @@ [activation (matmul x w +@ b); w_loss +@ b_loss]

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* [z; z_loss] = dense ~activation:tanh 784 512 x in
  let* [mean; mean_loss] = dense ~activation:Fun.id 512 embedding_dim z in
  let* [logvar; logvar_loss] = dense ~activation:Fun.id 512 embedding_dim z in
  return [mean; logvar; z_loss +@ mean_loss +@ logvar_loss]

let decoder z =
  let open Parameters in
  let* [z; z_loss] = dense ~activation:tanh embedding_dim 512 z in
  let* [z; z'_loss] = dense 512 784 z in
  return [z; z_loss +@ z'_loss]

let mse x x' = sum [0; 1] @@ mean [0] ((x -@ x') *@ (x -@ x'))

let vae x =
  let open Parameters in
  let* [mean'; logvar; encoder_loss] = encoder x in
  let z = reparametrize mean' (exp logvar) [batch_size; 1; embedding_dim] in
  let* [x'; decoder_loss] = decoder z in
  let kl = kl mean' logvar @@ exp logvar in
  let mse = mse x x' in
  return (mse +@ kl +@ encoder_loss +@ decoder_loss)

(* let optim = Optim.sgd 0.000001 *)

let optim = Optim.adamw ?lr:(Some 0.0001)

let train x = optim (vae x)

let train_compiled =
  let func =
    Parameters.create_func (Tensor_type ([batch_size; 1; 784], F32)) train
  in
  Ir.compile func

let reconstruct x =
  let open Parameters in
  let* [mean'; logvar; _] = encoder x in
  let z = reparametrize mean' logvar [batch_size; 1; embedding_dim] in
  let* x' = decoder z in
  return x'

let reconstruct_compiled =
  let func =
    Parameters.create_func
      (Tensor_type ([batch_size; 1; 784], F32))
      (fun x ->
        let open Parameters in
        let* [y; _] = reconstruct x in
        return y )
  in
  Ir.compile func

let decoder_compiled =
  let func =
    Parameters.create_func
      (Tensor_type ([batch_size; 1; embedding_dim], F32))
      (fun x ->
        let open Parameters in
        let* [y; _] = decoder x in
        return y )
  in
  Ir.compile func

let () =
  print_endline train_compiled ;
  Compile.compile train_compiled "train.vmfb" ;
  print_endline reconstruct_compiled ;
  Compile.compile reconstruct_compiled "reconstruct.vmfb" ;
  print_endline decoder_compiled ;
  Compile.compile decoder_compiled "decoder.vmfb"

let main =
  Backpropagate.diff Var (fun [a; b] ->
      matmul a (Ir.Var.BroadcastInDim (b, [4])) )

let main =
  Ir.create_func
    (List_type [Tensor_type ([4; 2; 1], F32); Tensor_type ([1; 2], F32)])
    main

let main_compiled = Ir.compile main

let () =
  print_endline main_compiled ;
  Compile.compile main_compiled "main.vmfb"
