open Dsl

let sigmoid x = 1. /.< (1. +.< exp (-1. *.< x))

let bayesian_parameter batch_size shape std_prior =
  let open Parameters in
  let* (E mean) = new_param (Runtime.HostValue.zeros (E (shape, F32))) in
  let* (E log_std) =
    new_param (E (Ir.Tensor.full F32 shape @@ Float.log std_prior))
  in
  return
  @@ Svi.sample
       ~prior:(Normal (zeros_like mean, ones_like mean *.> std_prior))
       ~guide:(Normal (mean, exp log_std))
       ~batch_size ()

let dense_bayesian ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let shape = Ir.shape_of_var x in
  let batch_size = List.hd shape in
  let* w = bayesian_parameter batch_size [in_dims; out_dims] 0.01 in
  let* b = bayesian_parameter batch_size [1; out_dims] 0.01 in
  return @@ activation (matmul x w +@ b)

let normal mean std =
  let u_1 = Stdlib.Random.float 1. in
  let u_2 = Stdlib.Random.float 1. in
  let r = Float.sqrt (-2. *. Float.log u_1) in
  let theta = 2. *. Float.pi *. u_2 in
  (r *. Float.sin theta *. std) +. mean

let normal_tensor mean std shape =
  let len = List.fold_left ( * ) 1 shape in
  let list = List.init len (fun _ -> normal mean std) in
  Ir.Tensor.of_list F32 shape list

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let shape = Ir.shape_of_var x in
  let batch_size = List.hd shape in
  let* (E w) =
    new_param (E (normal_tensor 0. 0.01 [batch_size; in_dims; out_dims]))
  in
  let* (E b) =
    new_param (E (normal_tensor 0. 0.01 [batch_size; 1; out_dims]))
  in
  return @@ activation (matmul x w +@ b)

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* z = dense ~activation:tanh 784 512 x in
  let* mean = dense ~activation:Fun.id 512 embedding_dim z in
  let* logstd = dense ~activation:Fun.id 512 embedding_dim z in
  return (mean, logstd)

let decoder z =
  let open Parameters in
  let* z = dense_bayesian ~activation:tanh embedding_dim 512 z in
  let* z = dense_bayesian 512 784 z in
  return (z *.> 100.)

let vae x =
  let open Parameters in
  let* mean', logstd = encoder x in
  let z =
    Svi.sample
      ~prior:(Normal (zeros_like mean', ones_like mean'))
      ~guide:(Normal (mean', exp logstd))
      ()
  in
  let* x' = decoder z in
  return @@ Distribution.Normal (x', ones_like x' *.> 0.02)

let optim = Optim.adamw ~lr:0.001

let train (Ir.Var.List.E x) = optim @@ Svi.elbo x @@ vae x

let decode Ir.Var.List.[] =
  let open Parameters in
  let x = norm (zeros ([], F32)) (ones ([], F32)) [1; 1; embedding_dim] in
  let* y = Svi.inference @@ decoder x in
  return @@ Ir.Var.List.E y
